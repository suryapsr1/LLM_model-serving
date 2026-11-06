import asyncio
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.engine.local import LocalLlamaEngine


class FakeLlama:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.started = threading.Event()
        self.resume = threading.Event()
        self.resume.set()
        self.closed = False
        self.error = False

    def __call__(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.set()
        try:
            if not self.resume.wait(3):
                raise TimeoutError('Test native call did not resume')
            time.sleep(0.02)
            if self.error:
                raise ValueError('native failure')
            yield {'choices': [{'text': 'hello', 'finish_reason': 'stop'}]}
        finally:
            self.active -= 1

    def close(self):
        self.closed = True


class LocalEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        model = Path(self.directory.name) / 'test.gguf'
        model.touch()
        self.native = FakeLlama()
        with patch('llama_cpp.Llama', return_value=self.native):
            self.engine = LocalLlamaEngine(str(model))

    def tearDown(self):
        self.native.resume.set()
        self.engine.close()
        self.directory.cleanup()

    async def consume(self):
        return [chunk async for chunk in self.engine.generate_stream('hello')]

    async def test_concurrent_streams_do_not_share_native_context(self):
        results = await asyncio.gather(self.consume(), self.consume(), self.consume())
        self.assertEqual(self.native.peak, 1)
        self.assertTrue(all(result[-1].is_eos for result in results))
        self.assertEqual(self.native.active, 0)

    async def test_cancellation_keeps_model_locked_until_native_work_finishes(self):
        self.native.resume.clear()
        first = asyncio.create_task(self.consume())
        for _ in range(100):
            if self.native.started.is_set():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(self.native.started.is_set())
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(self.consume())
        await asyncio.sleep(0.05)
        self.assertFalse(second.done())
        self.assertEqual(self.native.active, 1)
        self.native.resume.set()
        await asyncio.wait_for(second, 3)
        self.assertEqual(self.native.peak, 1)
        self.assertEqual(self.native.active, 0)

    async def test_native_error_propagates_and_releases_model(self):
        self.native.error = True
        with self.assertRaisesRegex(ValueError, 'native failure'):
            await self.consume()
        self.native.error = False
        self.assertEqual((await self.consume())[0].text, 'hello')

    async def test_early_stream_close_releases_model(self):
        stream = self.engine.generate_stream('hello')
        await anext(stream)
        await stream.aclose()
        self.assertEqual(self.native.active, 0)
        self.assertEqual((await self.consume())[0].text, 'hello')


if __name__ == '__main__':
    unittest.main()
