"""OpenAI-compatible chat completions client (OpenAI, vLLM, SGLang, OpenRouter)."""
import logging
from typing import Any, Dict

import httpx
from openai import AsyncOpenAI

from ..core.schemas import ModelResponse, TokenUsage
from ..utils.usage_normalizer import extract_openai_usage, to_token_usage
from .base import BaseModelClient, classify_empty_response
from .registry import register_provider

logger = logging.getLogger(__name__)


@register_provider("chat_completion")
class OpenAIClient(BaseModelClient):
    """Client for any OpenAI-compatible chat completions endpoint."""

    protected_paths = ("model", "messages", "stream", "stream_options")
    provider_name = "chat_completion"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # max_retries=0: BaseModelClient.generate owns retry; the SDK's own loop
        # would stack with ours. read timeout = per-chunk gap during streaming.
        client_kwargs = {
            'timeout': httpx.Timeout(connect=30.0, read=float(self.timeout), write=60.0, pool=30.0),
            'max_retries': 0,
        }
        if self.api_key:
            client_kwargs['api_key'] = self.api_key
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
            if 'api_key' not in client_kwargs:
                client_kwargs['api_key'] = 'dummy-key'

        self.client = AsyncOpenAI(**client_kwargs)

    async def aclose(self) -> None:
        await self.client.close()

    def build_request(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        request["max_tokens"] = max_output_tokens
        if self.top_p is not None:
            request["top_p"] = self.top_p
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
        return request

    async def _dispatch(self, request: Dict[str, Any]) -> ModelResponse:
        try:
            text = ""
            reasoning_chars = 0
            finish_reason = None
            model_name = self.model
            usage_chunk = None

            stream = await self.client.chat.completions.create(**request)
            async for chunk in stream:
                if chunk.model:
                    model_name = chunk.model
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        text += delta.content
                    if delta:
                        reasoning_delta = getattr(delta, "reasoning_content", None)
                        if reasoning_delta is None and getattr(delta, "model_extra", None):
                            reasoning_delta = delta.model_extra.get("reasoning_content")
                        if reasoning_delta:
                            reasoning_chars += len(reasoning_delta)
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                if chunk.usage:
                    usage_chunk = chunk

            if usage_chunk:
                tokens = self._extract_tokens(usage_chunk, text)
            else:
                tokens = TokenUsage(
                    prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                    output_tokens=0, total_tokens=0,
                )

            empty_error = classify_empty_response(
                text,
                output_tokens=tokens.output_tokens,
                reasoning_evidence=tokens.reasoning_tokens > 0 or reasoning_chars > 0,
                budget_exhausted=finish_reason == "length",
                detail=f"finish_reason={finish_reason or 'unknown'}",
            )

            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,
                model=model_name,
                # Provider-literal only: a stream that ends without a finish
                # chunk stays None so audits can't mistake it for a real stop.
                finish_reason=finish_reason,
                error=empty_error,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _extract_tokens(self, response, final_text: str = "") -> TokenUsage:
        return to_token_usage(extract_openai_usage(response.usage, final_text=final_text))
