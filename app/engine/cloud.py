import json
from typing import AsyncGenerator

import httpx

from .base import BaseLLMEngine, GenerationChunk


async def _sse_data(response: httpx.Response) -> AsyncGenerator[str, None]:
    """Join SSE data fields, accepting the optional space after the colon."""
    data_lines = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            if separator and value.startswith(" "):
                value = value[1:]
            data_lines.append(value)

    if data_lines:
        yield "\n".join(data_lines)


class CloudFallbackEngine(BaseLLMEngine):
    """Stream completions from a compatible remote chat endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> AsyncGenerator[GenerationChunk, None]:
        """Yield text and a terminal chunk, or raise if the request fails."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        status = response.status_code
                        if status in (401, 403):
                            guidance = "check the API key and provider permissions"
                        elif status == 404:
                            guidance = "check the provider base URL and model name"
                        elif status == 429:
                            guidance = "check provider quota and rate limits before retrying"
                        elif status >= 500:
                            guidance = "the provider is unavailable; retry later"
                        else:
                            guidance = "check the provider configuration and request parameters"
                        # Provider bodies can echo credentials or other private request data.
                        raise RuntimeError(f"Cloud provider returned HTTP {status}; {guidance}.")

                    async for data_str in _sse_data(response):
                        if not data_str.strip():
                            continue
                        if data_str.strip() == "[DONE]":
                            yield GenerationChunk(text="", is_eos=True, finish_reason="stop")
                            return

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            raise RuntimeError(
                                "Cloud provider returned invalid JSON in its event stream."
                            ) from None

                        if not isinstance(data, dict):
                            raise RuntimeError("Cloud provider returned an invalid stream event.")
                        if "error" in data:
                            raise RuntimeError(
                                "Cloud provider reported a streaming error; check provider "
                                "availability, quota, and model configuration."
                            )

                        choices = data.get("choices")
                        if not isinstance(choices, list):
                            raise RuntimeError("Cloud provider returned invalid stream choices.")
                        # Usage and moderation events can have no completion choices.
                        if not choices:
                            continue
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            raise RuntimeError("Cloud provider returned an invalid stream choice.")

                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None and not isinstance(finish_reason, str):
                            raise RuntimeError("Cloud provider returned an invalid finish reason.")
                        if "delta" not in choice and finish_reason is None:
                            raise RuntimeError("Cloud provider returned a stream choice without a delta.")
                        delta = choice.get("delta")
                        if delta is None:
                            delta = {}
                        if not isinstance(delta, dict):
                            raise RuntimeError("Cloud provider returned an invalid stream delta.")
                        token_text = delta.get("content")
                        if token_text is None:
                            token_text = ""
                        if not isinstance(token_text, str):
                            raise RuntimeError("Cloud provider returned invalid text content.")

                        is_eos = finish_reason is not None
                        if token_text or is_eos:
                            yield GenerationChunk(
                                text=token_text,
                                is_eos=is_eos,
                                finish_reason=finish_reason,
                            )
                        if is_eos:
                            return

                    raise RuntimeError(
                        "Cloud provider stream ended before a completion marker was received; retry."
                    )
        except httpx.TimeoutException:
            raise RuntimeError(
                "Cloud provider request timed out; retry or check provider availability."
            ) from None
        except httpx.RequestError:
            raise RuntimeError(
                "Could not reach cloud provider; check connectivity and the provider base URL."
            ) from None

                
