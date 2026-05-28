"""OpenAI-compatible chat completions client (OpenAI, vLLM, SGLang, OpenRouter)."""
import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI

from ..core.schemas import ModelResponse, TokenUsage
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

        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if self._is_reasoning_model():
            request_params["max_completion_tokens"] = max_output_tokens
            if kwargs.get("reasoning_effort"):
                request_params["reasoning_effort"] = kwargs["reasoning_effort"]
            request_params.pop("temperature", None)
            for param in ["top_p", "top_k", "repeat_penalty", "frequency_penalty", "presence_penalty"]:
                kwargs.pop(param, None)
        else:
            request_params["max_tokens"] = max_output_tokens
            if kwargs.get("top_p") is not None:
                request_params["top_p"] = kwargs["top_p"]
            if kwargs.get("top_k") is not None:
                request_params["top_k"] = kwargs["top_k"]
            if kwargs.get("frequency_penalty") is not None:
                request_params["frequency_penalty"] = kwargs["frequency_penalty"]
            if kwargs.get("presence_penalty") is not None:
                request_params["presence_penalty"] = kwargs["presence_penalty"]

        base_url_lower = (self.base_url or "").lower()
        is_deepseek_direct = "deepseek.com" in base_url_lower
        is_mimo_direct = "xiaomimimo.com" in base_url_lower
        is_openrouter = "openrouter" in base_url_lower

        if is_mimo_direct:
            request_params["max_completion_tokens"] = request_params.pop("max_tokens", max_output_tokens)
            if kwargs.get("enable_thinking") is not None:
                thinking_type = "enabled" if kwargs["enable_thinking"] else "disabled"
                request_params["extra_body"] = {"thinking": {"type": thinking_type}}
        elif is_deepseek_direct:
            # DeepSeek native API: {"thinking": {"type": "enabled"|"disabled"}}.
            # Thinking is enabled by default; strip params the docs say are ignored
            # in thinking mode (temperature, top_p, presence_penalty, frequency_penalty).
            explicit_thinking = kwargs.get("enable_thinking")
            thinking_on = explicit_thinking is not False
            if explicit_thinking is not None:
                thinking_type = "enabled" if explicit_thinking else "disabled"
                request_params["extra_body"] = {"thinking": {"type": thinking_type}}
            if kwargs.get("reasoning_effort"):
                eb = request_params.setdefault("extra_body", {})
                eb["reasoning_effort"] = kwargs["reasoning_effort"]
            if thinking_on:
                for p in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                    request_params.pop(p, None)
        elif is_openrouter:
            if kwargs.get("enable_thinking"):
                request_params["extra_body"] = {"reasoning": {"enabled": True}}
            elif kwargs.get("reasoning_effort") and not self._is_reasoning_model():
                request_params["extra_body"] = {"reasoning": {"effort": kwargs["reasoning_effort"]}}
        elif kwargs.get("enable_thinking") is not None:
            # vLLM/SGLang local: chat_template_kwargs
            thinking_key = "thinking" if "deepseek" in self.model.lower() else "enable_thinking"
            request_params["extra_body"] = {
                "chat_template_kwargs": {thinking_key: kwargs["enable_thinking"]}
            }

        request_params["stream"] = True
        request_params["stream_options"] = {"include_usage": True}

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

    def _is_reasoning_model(self) -> bool:
        model_lower = self.model.lower()
        return any(prefix in model_lower for prefix in ['o1', 'o3', 'o4', 'gpt-5'])
