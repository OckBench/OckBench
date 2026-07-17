"""Anthropic Messages API client with streaming and adaptive thinking."""
import json
import logging
from typing import Any, Dict, Tuple

import httpx

from ..core.schemas import ModelResponse
from ..utils.usage_normalizer import normalize_anthropic_usage, to_token_usage
from .base import BaseModelClient, classify_empty_response, raise_status_error
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

        # Anthropic-compatible relays may serve /messages but not
        # /messages/count_tokens. Once that route returns a definitive
        # "unsupported" status, remember it for the rest of this client's life
        # instead of re-probing (and re-warning) on every problem.
        self._count_tokens_supported = True

        # httpx does not auto-retry; BaseModelClient.generate is the sole retry owner.
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=30.0, read=self.timeout),
        )

    async def aclose(self) -> None:
        await self._http_client.aclose()

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
            thinking_tokens = None
            cache_creation_tokens = 0
            cache_read_tokens = 0
            cache_ephemeral_5m = 0
            cache_ephemeral_1h = 0
            model_name = self.model
            stop_reason = None
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
                                # The official API reports the exact thinking
                                # count here; compatible relays may omit it.
                                details = usage.get("output_tokens_details") or {}
                                if details.get("thinking_tokens") is not None:
                                    thinking_tokens = details["thinking_tokens"]
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

            # The normalization seam takes the provider's exact thinking count
            # verbatim when the stream reported it; only when the field is
            # absent (compatible relays) does it derive answer_tokens by
            # running the visible answer text through the injected counter and
            # get reasoning_tokens by subtraction. Cache metrics are logged only.
            normalized = await normalize_anthropic_usage(
                prompt_tokens=input_tokens,
                output_tokens=output_tokens,
                final_text=text,
                count_tokens=self._count_tokens_exact,
                thinking_tokens=thinking_tokens,
                cache_metrics={
                    "cache_creation_tokens": cache_creation_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_ephemeral_5m": cache_ephemeral_5m,
                    "cache_ephemeral_1h": cache_ephemeral_1h,
                },
            )
            tokens = to_token_usage(normalized)

            # A 200 stream can still be a degenerate non-answer (observed: a
            # relay ends the stream as end_turn with no content blocks and zero
            # usage). Anthropic bills thinking inside output_tokens, so with an
            # empty answer the normalized reasoning count equals output_tokens.
            error = classify_empty_response(
                text,
                output_tokens=output_tokens,
                reasoning_evidence=tokens.reasoning_tokens > 0,
                budget_exhausted=stop_reason == "max_tokens",
                detail=f"stop_reason={stop_reason}, output_tokens={output_tokens}",
            )

            return ModelResponse(
                text=text, tokens=tokens, latency=0, model=model_name,
                finish_reason=stop_reason, error=error,
            )

        except httpx.HTTPStatusError as e:
            error_body = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"Anthropic API HTTP error: {e.response.status_code} - {error_body}")
            # Carry status_code so the base layer can classify non-retryable errors.
            raise_status_error(e.response.status_code, error_body)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _count_tokens_exact(self, text: str) -> Tuple[int, bool]:
        """Return ``(count, estimated)`` for ``text`` via count_tokens.

        ``estimated`` is False when the count came from Anthropic's exact
        count_tokens endpoint (which includes a few tokens of user-message
        wrapping overhead (~4-6) — noise (<1%) for substantive answers,
        self-correcting via min(answer, output) clamping for very short ones)
        and True when it came from the tiktoken-based fallback. A definitive
        unsupported status (404/405/501) disables the endpoint for the rest of
        this client's life with a single warning; transient failures fall back
        per-call and keep probing.
        """
        if not text:
            return 0, False
        if self._count_tokens_supported:
            try:
                resp = await self._http_client.post(
                    self.count_tokens_url,
                    headers=self.headers,
                    json={"model": self.model, "messages": [{"role": "user", "content": text}]},
                    timeout=30.0,
                )
                resp.raise_for_status()
                return int(resp.json().get("input_tokens", 0)), False
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (404, 405, 501):
                    self._count_tokens_supported = False
                    logger.warning(
                        f"count_tokens endpoint unsupported (HTTP {e.response.status_code}); "
                        "using tiktoken estimates for the answer-token split for the rest of this run"
                    )
                else:
                    logger.warning(f"count_tokens failed, falling back to tiktoken estimate: {e}")
            except Exception as e:
                logger.warning(f"count_tokens failed, falling back to tiktoken estimate: {e}")
        from ..utils.token_counter import estimate_tokens
        return estimate_tokens(text, self.model), True
