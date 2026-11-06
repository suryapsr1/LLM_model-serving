from typing import Literal

from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    """Validate generation parameters before submitting work to an engine."""

    prompt: str = Field(..., min_length=1, description="User instruction or prompt string.")
    max_tokens: int = Field(default=128, ge=1, le=2048, description="Maximum tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling randomness scale.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus filtering threshold.")
    engine: Literal["local", "cloud"] = Field(default="local", description="Engine selection.")


class HealthCheckResponse(BaseModel):
    status: str
    active_engine: str
    version: str
