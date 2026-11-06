import os
from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Async LLM Serving Gateway"
    VERSION: str = "1.0.0"
    
    # Engine Settings
    LOCAL_MODEL_PATH: str = os.getenv(
        "LOCAL_MODEL_PATH", 
        str(Path(__file__).resolve().parents[2] / "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    )
    LOCAL_N_CTX: int = int(os.getenv("LOCAL_N_CTX", "2048"))
    LOCAL_N_THREADS: int = int(os.getenv("LOCAL_N_THREADS", "4"))
    DEFAULT_MAX_TOKENS: int = 128
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_TOP_P: float = 0.9
    
    # OpenAI-compatible cloud fallback settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

settings = Settings()
