"""Run a real local-model smoke test: python test_local.py."""

import asyncio
import sys
from contextlib import aclosing
from time import monotonic

from app.core.config import settings
from app.engine.local import LocalLlamaEngine


async def run_test():
    started = monotonic()
    engine = LocalLlamaEngine(
        model_path=settings.LOCAL_MODEL_PATH,
        n_ctx=settings.LOCAL_N_CTX,
        n_threads=settings.LOCAL_N_THREADS,
    )
    output = []
    saw_eos = False
    finish_reason = None
    try:
        print("Local model output: ", end="", flush=True)
        async with aclosing(engine.generate_stream(
            prompt="Say hello in one short sentence.",
            max_tokens=32,
            temperature=0.0,
            top_p=0.9,
        )) as stream:
            async for chunk in stream:
                reason = chunk.finish_reason or ""
                if reason.startswith(("error", "cloud_error_")):
                    raise RuntimeError(reason)
                output.append(chunk.text)
                print(chunk.text, end="", flush=True)
                if chunk.is_eos:
                    saw_eos = True
                    finish_reason = reason
                    break
        if not saw_eos:
            raise RuntimeError("Local generation ended without an EOS event.")
        if not "".join(output).strip():
            raise RuntimeError("Local generation returned no text.")
    finally:
        engine.close()
    print(f"\nPASS: local inference finished ({finish_reason}) in {monotonic() - started:.2f}s.")


if __name__ == "__main__":
    try:
        asyncio.run(run_test())
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
