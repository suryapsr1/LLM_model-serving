import json
import time
from contextlib import aclosing
from uuid import UUID

from celery import states
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from app.core.metrics import ACTIVE_STREAMS, OUTPUT_CHUNKS, TIME_TO_FIRST_CHUNK, TOTAL_GENERATION_DURATION
from app.schemas.inference import ChatCompletionRequest
from app.worker.celery_app import celery_app
from app.worker.tasks import batch_inference_task

router = APIRouter()


@router.post("/chat/stream")
async def stream_chat(request: ChatCompletionRequest, http_request: Request):
    engines = http_request.app.state.engines
    selected_engine = engines.get(request.engine)
    initial_engine = request.engine
    should_fallback = request.engine == "local" and selected_engine is not None
    if selected_engine is None and request.engine == "local":
        selected_engine = engines.get("cloud")
        initial_engine = "cloud"
    if selected_engine is None:
        raise HTTPException(503, f"Engine '{request.engine}' is unavailable or unconfigured.")

    def encode(payload):
        return f"data: {json.dumps(payload)}\n\n"

    async def events(engine):
        async with aclosing(engine.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )) as stream:
            async for chunk in stream:
                if chunk.is_eos and (chunk.finish_reason or "").startswith(("error:", "cloud_error_")):
                    raise RuntimeError(chunk.finish_reason)
                yield chunk
                if chunk.is_eos:
                    return
        raise RuntimeError("Generation ended without a terminal event.")

    async def event_publisher():
        start_time = time.perf_counter()
        first_token_emitted = False
        active_engine = initial_engine
        ACTIVE_STREAMS.inc()

        has_streamed_text = False
        try:
            async with aclosing(events(selected_engine)) as stream:
                async for chunk in stream:
                    if chunk.text:
                        if not first_token_emitted:
                            TIME_TO_FIRST_CHUNK.labels(engine=active_engine).observe(time.perf_counter() - start_time)
                            first_token_emitted = True
                        OUTPUT_CHUNKS.labels(engine=active_engine).inc()

                    has_streamed_text = has_streamed_text or bool(chunk.text)
                    yield encode(chunk.model_dump())

        except Exception as exc:
            cloud = engines.get("cloud") if should_fallback and not has_streamed_text else None
            if cloud is not None:
                active_engine = "cloud"
                try:
                    async with aclosing(events(cloud)) as stream:
                        async for chunk in stream:
                            if chunk.text:
                                if not first_token_emitted:
                                    TIME_TO_FIRST_CHUNK.labels(engine=active_engine).observe(time.perf_counter() - start_time)
                                    first_token_emitted = True
                                OUTPUT_CHUNKS.labels(engine=active_engine).inc()

                            yield encode(chunk.model_dump())
                    return
                except Exception as fallback_error:
                    exc = fallback_error

            yield encode({"error": str(exc), "is_eos": True, "finish_reason": "error"})

        finally:
            ACTIVE_STREAMS.dec()
            TOTAL_GENERATION_DURATION.labels(engine=active_engine).observe(time.perf_counter() - start_time)

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.post("/batch/submit", status_code=status.HTTP_202_ACCEPTED)
def submit_batch_task(request: ChatCompletionRequest):
    # Celery's Redis client is synchronous; FastAPI runs this handler in a thread.
    try:
        task = batch_inference_task.apply_async(kwargs=request.model_dump(), retry=False)
    except (OperationalError, RedisError, OSError, RuntimeError) as exc:
        raise HTTPException(503, "Batch queue is unavailable. Try again later.") from exc
    return {
        "status": "QUEUED",
        "task_id": task.id,
        "check_status_url": f"/v1/batch/status/{task.id}",
    }


@router.get("/batch/status/{task_id}")
def get_batch_status(task_id: UUID):
    try:
        result = celery_app.AsyncResult(str(task_id))
        state = result.state
        output = result.result if state in states.READY_STATES else None
    except (OperationalError, RedisError, OSError, RuntimeError) as exc:
        raise HTTPException(503, "Batch result store is unavailable. Try again later.") from exc
    response = {"task_id": str(task_id), "state": state, "result": None}
    if state in states.EXCEPTION_STATES:
        response["error"] = str(output)
    elif state == states.SUCCESS:
        response["result"] = output
    return response
