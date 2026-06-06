"""Anthropic Messages API client with streaming and adaptive thinking."""
import json
import logging
from typing import Any, Dict

import httpx

from ..core.schemas import ModelResponse
from ..utils.usage_normalizer import normalize_anthropic_usage, to_token_usage
from .base import BaseModelClient
from .registry import register_provider

logger = logging.getLogger(__name__)


@register_provider("anthropic")
class AnthropicClient(BaseModelClient):
    """Client for Anthropic /v1/messages with adaptive thinking and streaming.

    The default ``thinking`` block (``display="summarized"``) keeps the stream
    from going silent for minutes, but it is an ordinary, overridable request
    field — not hard-coded reasoning placement. Users reshape reasoning the same
    way as on every other provider (e.g. set ``output_config.effort`` or change
    ``thinking``).
    """

    protected_paths = ("model", "messages", "stream")
    provider_name = "anthropic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.endpoint = (self.base_url or "https://api.anthropic.com").rstrip("/")
        if self.endpoint.endswith("/v1"):
            self.messages_url = self.endpoint + "/messages"
            self.count_tokens_url = self.endpoint + "/messages/count_tokens"
        else:
            self.messages_url = self.endpoint + "/v1/messages"
            self.count_tokens_url = self.endpoint + "/v1/messages/count_tokens"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'dummy-key'}",
            "x-api-key": self.api_key or "dummy-key",
            "anthropic-version": "2023-06-01",
        }

        # httpx does not auto-retry; BaseModelClient.generate is the sole retry owner.
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=30.0, read=self.timeout),
        )

    def build_request(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        # display="summarized" forces thinking_delta events during reasoning so
        # the stream doesn't go silent. Summarized thinking text is not used for
        # token accounting (see _count_tokens_exact), only as a heartbeat.
        return {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive", "display": "summarized"},
            "stream": True,
        }

    async def _dispatch(self, request: Dict[str, Any]) -> ModelResponse:
        try:
            text = ""
            input_tokens = 0
            output_tokens = 0
            cache_creation_tokens = 0
            cache_read_tokens = 0
            cache_ephemeral_5m = 0
            cache_ephemeral_1h = 0
            model_name = self.model
            stop_reason = "end_turn"
            buffer = ""

            async with self._http_client.stream(
                "POST", self.messages_url, headers=self.headers, json=request
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp,
                    )

                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            d = json.loads(data_str)
                            event_type = d.get("type")

                            if event_type == "message_start":
                                msg = d.get("message", {})
                                usage = msg.get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)
                                cache_creation_tokens = usage.get("cache_creation_input_tokens", 0) or 0
                                cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
                                creation = usage.get("cache_creation") or {}
                                cache_ephemeral_5m = creation.get("ephemeral_5m_input_tokens", 0) or 0
                                cache_ephemeral_1h = creation.get("ephemeral_1h_input_tokens", 0) or 0
                                model_name = msg.get("model", self.model)

                            elif event_type == "content_block_delta":
                                delta = d.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text += delta.get("text", "")

                            elif event_type == "message_delta":
                                # usage in message_delta is cumulative; replace, don't add.
                                usage = d.get("usage", {})
                                output_tokens = usage.get("output_tokens", output_tokens)
                                cache_creation_tokens = (
                                    usage.get("cache_creation_input_tokens", cache_creation_tokens)
                                    or cache_creation_tokens
                                )
                                cache_read_tokens = (
                                    usage.get("cache_read_input_tokens", cache_read_tokens)
                                    or cache_read_tokens
                                )
                                stop_reason = d.get("delta", {}).get("stop_reason", stop_reason)

                        except json.JSONDecodeError:
                            continue

            # Anthropic reports a combined output_tokens (thinking + answer). The
            # normalization seam derives exact answer_tokens by running the
            # visible answer text through the injected counter, then gets
            # reasoning_tokens by subtraction. Cache metrics are logged only.
            normalized = await normalize_anthropic_usage(
                prompt_tokens=input_tokens,
                output_tokens=output_tokens,
                final_text=text,
                count_tokens=self._count_tokens_exact,
                cache_metrics={
                    "cache_creation_tokens": cache_creation_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_ephemeral_5m": cache_ephemeral_5m,
                    "cache_ephemeral_1h": cache_ephemeral_1h,
                },
            )
            tokens = to_token_usage(normalized)

            return ModelResponse(
                text=text, tokens=tokens, latency=0, model=model_name,
                finish_reason=stop_reason,
            )

        except httpx.HTTPStatusError as e:
            error_body = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"Anthropic API HTTP error: {e.response.status_code} - {error_body}")
            # Carry status_code so the base layer can classify non-retryable errors.
            wrapped = Exception(f"Error code: {e.response.status_code} - {error_body}")
            wrapped.status_code = e.response.status_code
            raise wrapped
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _count_tokens_exact(self, text: str) -> int:
        """Return Anthropic's exact token count for ``text`` via count_tokens.

        Includes a few tokens of user-message wrapping overhead (~4-6), which is
        noise (<1%) for substantive answers and self-corrects via min(answer,
        output) clamping for very short answers. Falls back to a tiktoken-based
        estimate if the endpoint is unavailable.
        """
        if not text:
            return 0
        try:
            resp = await self._http_client.post(
                self.count_tokens_url,
                headers=self.headers,
                json={"model": self.model, "messages": [{"role": "user", "content": text}]},
                timeout=30.0,
            )
            resp.raise_for_status()
            return int(resp.json().get("input_tokens", 0))
        except Exception as e:
            logger.warning(f"count_tokens failed, falling back to tiktoken estimate: {e}")
            from ..utils.token_counter import estimate_tokens
            return estimate_tokens(text, self.model)
