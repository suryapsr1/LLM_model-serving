import json
import time

from locust import HttpUser, between, task


class LLMGatewayUser(HttpUser):
    # Simulated pause between queries (1 to 2 seconds)
    wait_time = between(1, 2)

    @task(3)
    def test_health(self):
        """Standard liveness probe."""
        self.client.get("/healthz")

    @task(2)
    def test_streaming(self):
        """Simulate real-time streaming queries."""
        payload = {
            "prompt": "Explain asynchronous programming in one sentence.",
            "max_tokens": 30,
            "engine": "local"
        }
        started = time.perf_counter()
        with self.client.post(
            "/v1/chat/stream",
            json=payload,
            stream=True,
            catch_response=True,
        ) as response:
            try:
                if response.status_code != 200:
                    response.failure(response.error or f"HTTP {response.status_code}")
                    return

                saw_text = False
                saw_end = False
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if not line.startswith("data:"):
                            continue
                        event = json.loads(line[5:].strip())
                        if event.get("error"):
                            response.failure(f"Stream error: {event['error']}")
                            return
                        if (event.get("finish_reason") or "").startswith(("error:", "cloud_error_")):
                            response.failure(f"Stream error: {event['finish_reason']}")
                            return
                        saw_text = saw_text or bool(event.get("text"))
                        if event.get("is_eos"):
                            saw_end = True
                            break
                except (UnicodeError, ValueError, AttributeError, TypeError) as exc:
                    response.failure(f"Invalid stream response: {exc}")
                    return

                if not saw_end or not saw_text:
                    response.failure("Stream ended without a complete text response")
                else:
                    response.success()
            finally:
                # Locust's stream=True timing stops when headers arrive.
                response.request_meta["response_time"] = (time.perf_counter() - started) * 1000

    @task(1)
    def test_batch_enqueue(self):
        """Simulate batch job submission."""
        payload = {
            "prompt": "Summarize the history of computing.",
            "max_tokens": 50,
            "engine": "local"
        }
        with self.client.post("/v1/batch/submit", json=payload, catch_response=True) as response:
            if response.status_code != 202:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                job = response.json()
            except ValueError:
                response.failure("Batch submission returned invalid JSON")
                return
            if not isinstance(job, dict) or job.get("status") != "QUEUED" or not job.get("task_id"):
                response.failure("Batch submission did not return a queued task ID")
            else:
                response.success()
