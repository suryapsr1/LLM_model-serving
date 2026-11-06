import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from celery import states
from fastapi.testclient import TestClient
from kombu.exceptions import OperationalError
from prometheus_client import REGISTRY
from redis.exceptions import RedisError

from app import main
from app.api import routes
from app.engine.base import GenerationChunk


class StubEngine:
    def __init__(self, *items):
        self.items = items
        self.calls = []
        self.closed = False

    async def generate_stream(self, **kwargs):
        self.calls.append(kwargs)
        try:
            for item in self.items:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.closed = True


class RouteTests(unittest.TestCase):
    def setUp(self):
        engine_patch = patch.dict(main.engines, {}, clear=True)
        engine_patch.start()
        self.addCleanup(engine_patch.stop)
        state_patch = patch.object(main.app.state, "engines", main.engines)
        state_patch.start()
        self.addCleanup(state_patch.stop)
        # TestClient runs lifespan only when entered as a context manager.
        # All engines and Celery calls here are in-memory test doubles.
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)
        self.task_id = str(uuid4())

    def events(self, **overrides):
        payload = {"prompt": "hello", **overrides}
        response = self.client.post("/v1/chat/stream", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        return [json.loads(line[5:].strip()) for line in response.text.splitlines()
                if line.startswith("data:")]

    def test_health_without_engines_is_unavailable(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "unavailable")
        self.assertEqual(response.json()["active_engine"], "unavailable")

    def test_health_reports_available_engine(self):
        for engine_name in ("local", "cloud"):
            with self.subTest(engine=engine_name):
                main.engines.clear()
                main.engines[engine_name] = StubEngine()
                response = self.client.get("/healthz")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "healthy")
                self.assertEqual(response.json()["active_engine"], engine_name)

    def test_invalid_and_null_parameters_are_rejected(self):
        invalid = (
            {"prompt": None}, {"prompt": ""}, {"max_tokens": None},
            {"max_tokens": 0}, {"max_tokens": 2049}, {"temperature": None},
            {"temperature": -0.1}, {"temperature": 2.1}, {"top_p": None},
            {"top_p": -0.1}, {"top_p": 1.1}, {"engine": None}, {"engine": "other"},
        )
        with patch.object(routes.batch_inference_task, "apply_async") as enqueue:
            for endpoint in ("/v1/chat/stream", "/v1/batch/submit"):
                for fields in invalid:
                    with self.subTest(endpoint=endpoint, fields=fields):
                        response = self.client.post(endpoint, json={"prompt": "hello", **fields})
                        self.assertEqual(response.status_code, 422, response.text)
            enqueue.assert_not_called()

    def test_unconfigured_stream_returns_service_unavailable(self):
        response = self.client.post("/v1/chat/stream", json={"prompt": "hello"})
        self.assertEqual(response.status_code, 503)

    def test_stream_forwards_parameters_and_closes_generator(self):
        local = StubEngine(
            GenerationChunk(text="Hi", is_eos=False),
            GenerationChunk(text="!", is_eos=True, finish_reason="length"),
        )
        main.engines["local"] = local
        events = self.events(max_tokens=9, temperature=0.2, top_p=0.4)
        self.assertEqual("".join(event["text"] for event in events), "Hi!")
        self.assertEqual(events[-1]["finish_reason"], "length")
        self.assertEqual(local.calls, [{
            "prompt": "hello", "max_tokens": 9, "temperature": 0.2, "top_p": 0.4,
        }])
        self.assertTrue(local.closed)

    def test_missing_local_engine_uses_cloud(self):
        cloud = StubEngine(GenerationChunk(text="cloud", is_eos=True, finish_reason="stop"))
        main.engines["cloud"] = cloud
        self.assertEqual(self.events()[-1]["text"], "cloud")
        self.assertEqual(len(cloud.calls), 1)

    def test_metrics_count_cloud_chunks_when_local_is_unavailable(self):
        cloud = StubEngine(
            GenerationChunk(text="Hello ", is_eos=False),
            GenerationChunk(text="world", is_eos=True, finish_reason="stop"),
        )
        main.engines["cloud"] = cloud

        def sample(name, engine=None):
            labels = {"engine": engine} if engine else None
            return REGISTRY.get_sample_value(name, labels) or 0

        cloud_chunks = sample("llm_output_chunks_total", "cloud")
        local_chunks = sample("llm_output_chunks_total", "local")
        cloud_first_chunk = sample("llm_time_to_first_chunk_seconds_count", "cloud")
        cloud_duration = sample("llm_generation_duration_seconds_count", "cloud")
        active_streams = sample("llm_active_streams")

        self.assertEqual("".join(event["text"] for event in self.events()), "Hello world")
        self.assertEqual(sample("llm_output_chunks_total", "cloud"), cloud_chunks + 2)
        self.assertEqual(sample("llm_output_chunks_total", "local"), local_chunks)
        self.assertEqual(sample("llm_time_to_first_chunk_seconds_count", "cloud"), cloud_first_chunk + 1)
        self.assertEqual(sample("llm_generation_duration_seconds_count", "cloud"), cloud_duration + 1)
        self.assertEqual(sample("llm_active_streams"), active_streams)
        metrics_response = self.client.get("/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        self.assertIn('llm_output_chunks_total{engine="cloud"}', metrics_response.text)

    def test_local_failure_before_text_uses_cloud(self):
        local = StubEngine(RuntimeError("local unavailable"))
        cloud = StubEngine(GenerationChunk(text="cloud", is_eos=True, finish_reason="stop"))
        main.engines.update(local=local, cloud=cloud)
        events = self.events()
        self.assertEqual([event["text"] for event in events], ["cloud"])
        self.assertTrue(local.closed)
        self.assertTrue(cloud.closed)

    def test_partial_local_output_prevents_cloud_fallback(self):
        local = StubEngine(GenerationChunk(text="partial", is_eos=False), RuntimeError("failed"))
        cloud = StubEngine(GenerationChunk(text="cloud", is_eos=True, finish_reason="stop"))
        main.engines.update(local=local, cloud=cloud)
        events = self.events()
        self.assertEqual(events[0]["text"], "partial")
        self.assertEqual(events[-1]["error"], "failed")
        self.assertTrue(events[-1]["is_eos"])
        self.assertEqual(events[-1]["finish_reason"], "error")
        self.assertEqual(cloud.calls, [])
        self.assertTrue(local.closed)

    def test_explicit_cloud_request_never_uses_local(self):
        local = StubEngine(GenerationChunk(text="local", is_eos=True, finish_reason="stop"))
        main.engines["local"] = local
        response = self.client.post("/v1/chat/stream", json={"prompt": "hello", "engine": "cloud"})
        self.assertEqual(response.status_code, 503)
        main.engines["cloud"] = StubEngine(RuntimeError("cloud failed"))
        events = self.events(engine="cloud")
        self.assertEqual(events[-1]["error"], "cloud failed")
        self.assertTrue(events[-1]["is_eos"])
        self.assertEqual(local.calls, [])

    def test_failed_fallback_emits_terminal_error(self):
        main.engines.update(
            local=StubEngine(RuntimeError("local failed")),
            cloud=StubEngine(RuntimeError("cloud failed")),
        )
        events = self.events()
        self.assertEqual(events, [{"error": "cloud failed", "is_eos": True, "finish_reason": "error"}])

    def test_stream_without_terminal_chunk_emits_error(self):
        main.engines["local"] = StubEngine(GenerationChunk(text="partial", is_eos=False))
        events = self.events()
        self.assertEqual(events[0]["text"], "partial")
        self.assertTrue(events[-1]["is_eos"])
        self.assertIn("without a terminal", events[-1]["error"])

    def test_batch_accepted_and_forwards_top_p(self):
        payload = {"prompt": "batch prompt", "max_tokens": 12, "temperature": 0.2,
                   "top_p": 0.45, "engine": "cloud"}
        with patch.object(routes.batch_inference_task, "apply_async",
                          return_value=SimpleNamespace(id=self.task_id)) as enqueue:
            response = self.client.post("/v1/batch/submit", json=payload)
        self.assertEqual(response.status_code, 202)
        enqueue.assert_called_once_with(kwargs=payload, retry=False)
        self.assertEqual(response.json(), {
            "status": "QUEUED", "task_id": self.task_id,
            "check_status_url": f"/v1/batch/status/{self.task_id}",
        })

    def test_batch_broker_and_backend_errors_are_service_unavailable(self):
        failures = (
            OperationalError("broker unavailable"), RedisError("backend unavailable"),
            OSError("connection failed"),
            RuntimeError("Retry limit exceeded while trying to reconnect to the Celery redis result store backend."),
        )
        for failure in failures:
            with self.subTest(error=type(failure).__name__):
                with patch.object(routes.batch_inference_task, "apply_async", side_effect=failure):
                    response = self.client.post("/v1/batch/submit", json={"prompt": "hello"})
                self.assertEqual(response.status_code, 503, response.text)
                self.assertIn("queue is unavailable", response.json()["detail"])

    def test_status_uses_configured_celery_app(self):
        result = {"status": "SUCCESS", "output": "hello"}
        with patch.object(routes.celery_app, "AsyncResult",
                          return_value=SimpleNamespace(state=states.SUCCESS, result=result)) as get_result:
            response = self.client.get(f"/v1/batch/status/{self.task_id}")
        self.assertEqual(response.status_code, 200)
        get_result.assert_called_once_with(self.task_id)
        self.assertEqual(response.json()["result"], result)

    def test_failed_task_exception_is_serialized_as_string(self):
        with patch.object(routes.celery_app, "AsyncResult", return_value=SimpleNamespace(
            state=states.FAILURE, result=RuntimeError("inference failed")
        )):
            response = self.client.get(f"/v1/batch/status/{self.task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "task_id": self.task_id, "state": "FAILURE", "result": None,
            "error": "inference failed",
        })

    def test_pending_status_does_not_read_result(self):
        class PendingResult:
            state = states.PENDING

            @property
            def result(self):
                raise AssertionError("A pending result must not be read")

        with patch.object(routes.celery_app, "AsyncResult", return_value=PendingResult()):
            response = self.client.get(f"/v1/batch/status/{self.task_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"], "PENDING")
        self.assertIsNone(response.json()["result"])

    def test_status_rejects_invalid_uuid(self):
        with patch.object(routes.celery_app, "AsyncResult") as get_result:
            response = self.client.get("/v1/batch/status/not-a-uuid")
        self.assertEqual(response.status_code, 422)
        get_result.assert_not_called()

    def test_status_backend_errors_are_service_unavailable(self):
        failures = (
            RedisError("backend unavailable"),
            RuntimeError("Retry limit exceeded while trying to reconnect to the Celery redis result store backend."),
        )
        for failure in failures:
            with self.subTest(error=type(failure).__name__):
                with patch.object(routes.celery_app, "AsyncResult", side_effect=failure):
                    response = self.client.get(f"/v1/batch/status/{self.task_id}")
                self.assertEqual(response.status_code, 503, response.text)
                self.assertIn("result store is unavailable", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
