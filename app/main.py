from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, Response
from app.core.config import settings
from app.engine.local import LocalLlamaEngine
from app.engine.cloud import CloudFallbackEngine
from app.schemas.inference import HealthCheckResponse
from app.api.routes import router as api_router
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# State storage for our persistent singletons
engines = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Load model once into RAM before accepting traffic
    print(f"[*] Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    print("[*] Initializing Local Llama Engine...")
    try:
        engines["local"] = await asyncio.to_thread(
            LocalLlamaEngine,
            model_path=settings.LOCAL_MODEL_PATH,
            n_ctx=settings.LOCAL_N_CTX,
            n_threads=settings.LOCAL_N_THREADS,
        )
    except Exception as exc:
        engines["local"] = None
        print(f"[!] Local engine unavailable: {exc}")
    
    # 2. Initialize cloud fallback if an API key is available
    if settings.OPENAI_API_KEY:
        engines["cloud"] = CloudFallbackEngine(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model_name=settings.OPENAI_MODEL,
        )
    else:
        engines["cloud"] = None
        
    app.state.engines = engines
    try:
        yield
    finally:
        print("[*] Shutting down inference service...")
        local_engine = engines.get("local")
        if local_engine is not None:
            await asyncio.to_thread(local_engine.close)
        engines.clear()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Attach routes
app.state.engines = engines
app.include_router(api_router, prefix="/v1")

@app.get("/healthz", response_model=HealthCheckResponse)
async def health_check(response: Response):
    available = bool(engines.get("local") or engines.get("cloud"))
    if not available:
        response.status_code = 503
    return HealthCheckResponse(
        status="healthy" if available else "unavailable",
        active_engine=(
            "local"
            if engines.get("local")
            else "cloud"
            if engines.get("cloud")
            else "unavailable"
        ),
        version=settings.VERSION
    )

@app.get("/metrics")
def get_metrics():
    """Exposes Prometheus scrape metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
