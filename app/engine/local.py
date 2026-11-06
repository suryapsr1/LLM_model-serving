import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from .base import BaseLLMEngine, GenerationChunk


class LocalLlamaEngine(BaseLLMEngine):
    def __init__(
        self, model_path: str, n_ctx: int = 2048, n_threads: int | None = None
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # A missing native library should not prevent a cloud-only API from starting.
        from llama_cpp import Llama

        self.n_threads = n_threads or min(4, max(1, (os.cpu_count() or 4) - 2))
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")
        self._closed = False

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[GenerationChunk, None]:
        # llama.cpp mutates a shared context. Own it for the entire stream,
        # including native work still running after a client disconnects.
        while not self._lock.acquire(blocking=False):
            await asyncio.sleep(0.01)

        formatted_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        def completion():
            yield from self.llm(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stop=["<|eot_id|>", "<|end_of_text|>"],
            )

        iterator = completion()
        pending = None

        def release(_future=None):
            try:
                iterator.close()
            finally:
                self._lock.release()

        try:
            if self._closed:
                raise RuntimeError("Local engine is closed.")
            while True:
                pending = self._executor.submit(next, iterator, None)
                chunk = await asyncio.wrap_future(pending)
                pending = None
                if chunk is None:
                    raise RuntimeError("Local generation ended without a finish reason.")

                choice = chunk["choices"][0]
                finish_reason = choice.get("finish_reason")
                yield GenerationChunk(
                    text=choice["text"],
                    is_eos=finish_reason is not None,
                    finish_reason=finish_reason,
                )
                if finish_reason is not None:
                    return
        finally:
            if pending is not None and not pending.done():
                # This Future finishes only when native inference stops; cancelling
                # its asyncio wrapper cannot release the model early.
                pending.add_done_callback(release)
            else:
                release()

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.llm.close()
        self._executor.shutdown(wait=True)
