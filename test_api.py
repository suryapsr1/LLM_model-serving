"""Smoke-test a running API, Redis and worker: python test_api.py."""

import argparse
import asyncio
import json
import sys
from time import monotonic

import httpx


async def run_test(base_url: str):
    started = monotonic()
    payload = {
        "prompt": "Say hello in one short sentence.",
        "max_tokens": 32,
        "temperature": 0.0,
        "top_p": 0.9,
        "engine": "local",
    }
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        health_response = await client.get("/healthz")
        health_response.raise_for_status()
        health = health_response.json()
        if health.get("status") != "healthy" or health.get("active_engine") != "local":
            raise RuntimeError(f"A healthy local engine is required for this smoke test: {health}")
        print("PASS: API health reports the local engine.")

        output = []
        saw_eos = False
        async with asyncio.timeout(120):
            async with client.stream("POST", "/v1/chat/stream", json=payload) as response:
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise RuntimeError("Streaming endpoint did not return an SSE response.")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = json.loads(line[5:].strip())
                    reason = chunk.get("finish_reason") or ""
                    if chunk.get("error") or reason.startswith(("error", "cloud_error_")):
                        raise RuntimeError(f"Streaming failed: {chunk.get('error') or reason}")
                    output.append(chunk.get("text", ""))
                    if chunk.get("is_eos"):
                        saw_eos = True
                        break
        stream_text = "".join(output).strip()
        if not saw_eos or not stream_text:
            raise RuntimeError("Streaming must produce nonempty text and an EOS event.")
        print(f"PASS: streaming completed with text: {stream_text}")

        submission = await client.post("/v1/batch/submit", json=payload)
        submission.raise_for_status()
        if submission.status_code != 202:
            raise RuntimeError(f"Batch submission must return 202, received {submission.status_code}.")
        queued = submission.json()
        task_id = queued.get("task_id")
        if queued.get("status") != "QUEUED" or not task_id:
            raise RuntimeError(f"Invalid batch acknowledgement: {queued}")
        print(f"Batch task queued: {task_id}")

        async with asyncio.timeout(180):
            previous_state = None
            while True:
                response = await client.get(f"/v1/batch/status/{task_id}")
                response.raise_for_status()
                task = response.json()
                state = task.get("state")
                if state != previous_state:
                    print(f"Batch state: {state}")
                    previous_state = state
                if task.get("error") or state in {"FAILURE", "REVOKED"}:
                    raise RuntimeError(f"Batch inference failed: {task}")
                if state == "SUCCESS":
                    result = task.get("result")
                    if (
                        not isinstance(result, dict)
                        or result.get("status") != "SUCCESS"
                        or result.get("engine") != "local"
                        or not isinstance(result.get("output"), str)
                        or not result["output"].strip()
                    ):
                        raise RuntimeError(f"Batch result has no successful local output: {result}")
                    print(f"PASS: batch completed with text: {result['output'].strip()}")
                    break
                if state not in {"PENDING", "RECEIVED", "STARTED", "RETRY"}:
                    raise RuntimeError(f"Unexpected batch state: {state}")
                await asyncio.sleep(0.5)

    print(f"PASS: API, local streaming, Redis and worker verified in {monotonic() - started:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        asyncio.run(run_test(args.base_url.rstrip("/")))
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
