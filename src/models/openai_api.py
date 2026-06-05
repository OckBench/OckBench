"""OpenAI-compatible chat completions client (OpenAI, vLLM, SGLang, OpenRouter)."""
import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI

from ..core.schemas import ModelResponse, TokenUsage
from ..utils.request_overrides import apply_request_overrides
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseModelClient):
    """Client for any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        super().__init__(model, api_key, base_url, timeout, max_retries, **kwargs)

        # max_retries=0: BaseModelClient.generate owns retry; SDK's own loop
        # would stack with ours (up to 9 effective attempts).
        # read timeout = per-chunk gap in streaming, not total deadline.
        client_kwargs = {
            'timeout': httpx.Timeout(connect=30.0, read=float(timeout), write=60.0, pool=30.0),
            'max_retries': 0,
        }
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
            if 'api_key' not in client_kwargs:
                client_kwargs['api_key'] = 'dummy-key'

        self.client = AsyncOpenAI(**client_kwargs)

    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]

        # Uniform base request. Provider/model-specific shaping is no longer
        # hard-coded here; users control it through request overrides.
        request_params = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            request_params["temperature"] = temperature
        request_params["max_tokens"] = max_output_tokens
        if kwargs.get("top_p") is not None:
            request_params["top_p"] = kwargs["top_p"]
        request_params["stream"] = True
        request_params["stream_options"] = {"include_usage": True}

        request_params = apply_request_overrides(
            request_params,
            kwargs.get("request_overrides"),
            {"max_output_tokens": max_output_tokens},
        )

        try:
            text = ""
            reasoning_chars = 0
            finish_reason = None
            model_name = self.model
            usage_data = None

            stream = await self.client.chat.completions.create(**request_params)
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
                    usage_data = chunk

            if usage_data:
                tokens = self._extract_tokens(usage_data)
            else:
                tokens = TokenUsage(
                    prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                    output_tokens=0, total_tokens=0,
                )

            # Surface empty-text outcomes as errors so --cache resume will retry them.
            # Common on third-party chat-completion proxies + reasoning models: the whole budget is
            # spent on reasoning_content and the stream ends without ever emitting
            # a content delta.
            empty_error = None
            if not text:
                if finish_reason == "length":
                    suffix = " after reasoning_content stream" if reasoning_chars else ""
                    empty_error = (
                        "empty_response_length_finish: finish_reason=length with no content "
                        f"emitted{suffix} (likely reasoning consumed entire output budget)"
                    )
                elif tokens.reasoning_tokens > 0 or reasoning_chars > 0:
                    empty_error = (
                        "empty_response_reasoning_only: model emitted reasoning tokens but "
                        "no content (finish_reason="
                        f"{finish_reason or 'unknown'})"
                    )
                elif usage_data is None:
                    empty_error = "empty_response_no_stream: no usage and no content received"
                else:
                    empty_error = (
                        "empty_response_no_content: stream completed with usage but no content "
                        f"(finish_reason={finish_reason or 'unknown'})"
                    )

            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,
                model=model_name,
                finish_reason=finish_reason or "stop",
                error=empty_error,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _extract_tokens(self, response) -> TokenUsage:
        usage = response.usage
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)

        reasoning_tokens = 0
        if hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            if details and hasattr(details, 'reasoning_tokens'):
                reasoning_tokens = details.reasoning_tokens or 0

        answer_tokens = completion_tokens - reasoning_tokens

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=completion_tokens,
            total_tokens=getattr(usage, 'total_tokens', 0),
        )
