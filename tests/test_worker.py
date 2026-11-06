import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.engine.base import GenerationChunk
from app.worker.tasks import batch_inference_task, get_worker_engine, worker_engines


class FakeEngine:
    def __init__(self, events):
        self.events = events
        self.parameters = None
        self.closed = False

    async def generate_stream(self, **parameters):
        self.parameters = parameters
        try:
            for event in self.events:
                if isinstance(event, Exception):
                    raise event
                yield event
        finally:
            self.closed = True


class BatchInferenceTests(unittest.TestCase):
    def run_task(self, fake_engine, **parameters):
        with patch("app.worker.tasks.get_worker_engine", return_value=fake_engine):
            return batch_inference_task.apply(
                kwargs={"prompt": "Hello", **parameters}, throw=False
            )

    def test_generation_exception_marks_task_as_failed(self):
        engine = FakeEngine([RuntimeError("inference unavailable")])
        result = self.run_task(engine)
        self.assertEqual(result.state, "FAILURE")
        self.assertIn("inference unavailable", str(result.result))
        self.assertTrue(engine.closed)

    def test_error_chunks_mark_task_as_failed(self):
        for reason in ("error: invalid context", "cloud_error_401: invalid API key"):
            with self.subTest(reason=reason):
                engine = FakeEngine(
                    [GenerationChunk(text="", is_eos=True, finish_reason=reason)]
                )
                result = self.run_task(engine)
                self.assertEqual(result.state, "FAILURE")
                self.assertIn(reason, str(result.result))
                self.assertTrue(engine.closed)

    def test_partial_output_is_not_reported_as_success_after_error(self):
        engine = FakeEngine(
            [
                GenerationChunk(text="partial", is_eos=False),
                GenerationChunk(text="", is_eos=True, finish_reason="error: interrupted"),
            ]
        )
        self.assertEqual(self.run_task(engine).state, "FAILURE")
        self.assertTrue(engine.closed)

    def test_success_preserves_sampling_and_closes_stream_at_eos(self):
        engine = FakeEngine(
            [
                GenerationChunk(text="Hello ", is_eos=False),
                GenerationChunk(text="world", is_eos=True, finish_reason="length"),
                RuntimeError("must not read beyond EOS"),
            ]
        )
        result = self.run_task(
            engine, max_tokens=32, temperature=0.2, top_p=0.4, engine="cloud"
        )
        self.assertEqual(result.state, "SUCCESS")
        self.assertEqual(result.result["output"], "Hello world")
        self.assertEqual(result.result["engine"], "cloud")
        self.assertEqual(
            engine.parameters,
            {"prompt": "Hello", "max_tokens": 32, "temperature": 0.2, "top_p": 0.4},
        )
        self.assertTrue(engine.closed)

    def test_engine_initialization_failures_propagate(self):
        with patch(
            "app.worker.tasks.get_worker_engine", side_effect=ValueError("missing model")
        ):
            result = batch_inference_task.apply(kwargs={"prompt": "Hello"}, throw=False)
        self.assertEqual(result.state, "FAILURE")
        self.assertIn("missing model", str(result.result))

    def test_worker_reuses_engine_instance(self):
        settings = SimpleNamespace(
            LOCAL_MODEL_PATH="model.gguf", LOCAL_N_CTX=1024, LOCAL_N_THREADS=2
        )
        with patch.dict(worker_engines, {}, clear=True):
            with patch("app.worker.tasks.settings", settings), patch(
                "app.worker.tasks.LocalLlamaEngine"
            ) as constructor:
                first = get_worker_engine("local")
                second = get_worker_engine("local")
                self.assertIs(first, second)
                constructor.assert_called_once_with(
                    model_path="model.gguf", n_ctx=1024, n_threads=2
                )


if __name__ == "__main__":
    unittest.main()
