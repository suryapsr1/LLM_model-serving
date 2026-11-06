import asyncio
from contextlib import aclosing

from celery.utils.log import get_task_logger

from app.core.config import settings
from app.engine.cloud import CloudFallbackEngine
from app.engine.local import LocalLlamaEngine
from app.worker.celery_app import celery_app


logger = get_task_logger(__name__)

# Each Celery process keeps its model loaded across tasks.
worker_engines = {}


def get_worker_engine(engine_name: str):
    if engine_name not in worker_engines:
        if engine_name == "local":
            logger.info("Loading local model for worker process")
            worker_engines["local"] = LocalLlamaEngine(
                model_path=settings.LOCAL_MODEL_PATH,
                n_ctx=settings.LOCAL_N_CTX,
                n_threads=settings.LOCAL_N_THREADS,
            )
        elif engine_name == "cloud":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured for cloud worker.")
            worker_engines["cloud"] = CloudFallbackEngine(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model_name=settings.OPENAI_MODEL,
            )
        else:
            raise ValueError(f"Unknown engine: {engine_name}")

    return worker_engines[engine_name]


@celery_app.task(bind=True, name="tasks.batch_inference")
def batch_inference_task(
    self,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    engine: str = "local",
    top_p: float = 0.9,
):
    """Collect streamed output, allowing Celery to record failures correctly."""
    logger.info("Starting batch task %s", self.request.id)
    selected_engine = get_worker_engine(engine)

    async def run_inference():
        full_text = []
        stream = selected_engine.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # Closing on EOS also releases an engine lock or an upstream HTTP stream.
        async with aclosing(stream):
            async for chunk in stream:
                reason = chunk.finish_reason or ""
                if reason.startswith(("error:", "cloud_error_")):
                    raise RuntimeError(f"Generation failed: {reason}")
                full_text.append(chunk.text)
                if chunk.is_eos:
                    break
        return "".join(full_text)

    # Returning an error dictionary would incorrectly mark the task as SUCCESS.
    # Let generation and initialization exceptions reach Celery's failure handler.
    result_text = asyncio.run(run_inference())
    logger.info("Completed batch task %s", self.request.id)
    return {
        "status": "SUCCESS",
        "task_id": self.request.id,
        "engine": engine,
        "output": result_text,
    }
