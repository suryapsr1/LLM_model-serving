from abc import ABC, abstractmethod #a common interface (a set of methods and properties) that any child subclass must implement
from typing import AsyncGenerator
from pydantic import BaseModel, Field #validates data and provides a structured way to define data models


class GenerationChunk(BaseModel):
    text: str= Field(description = "The decoded string for current token")
    is_eos: bool = Field(description = "Whether the current token is the end of the sequence")
    finish_reason: str | None = Field( default = None,description = "Why generation stopped")



class BaseLLMEngine(ABC):
    """
    Base class for all LLM engines. This class defines the interface that all LLM engines must implement.
    """
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,

    ) -> AsyncGenerator[GenerationChunk, None]:
        """
        Generate text based on the given prompt and parameters.

        Args:
            prompt (str): The input prompt to generate text from.
            max_tokens (int): The maximum number of tokens to generate. Default is 256.
            temperature (float): The sampling temperature. Default is 0.7.
            top_p (float): The cumulative probability for nucleus sampling. Default is 0.9.
        """

        pass
    