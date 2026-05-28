"""Anthropic Messages API client with streaming and adaptive thinking."""
import json
import logging
from typing import Optional

import httpx

from ..core.schemas import ModelResponse, TokenUsage
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseModelClient):
    """Client for Anthropic /v1/messages with adaptive thinking and streaming."""

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

        self.endpoint = (base_url or "https://api.anthropic.com").rstrip("/")
        if self.endpoint.endswith("/v1"):
            self.messages_url = self.endpoint + "/messages"
            self.count_tokens_url = self.endpoint + "/messages/count_tokens"
        else:
            self.messages_url = self.endpoint + "/v1/messages"
            self.count_tokens_url = self.endpoint + "/v1/messages/count_tokens"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'dummy-key'}",
            "x-api-key": api_key or "dummy-key",
            "anthropic-version": "2023-06-01",
        }

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0, read=timeout),
        )

    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        # display="summarized" forces thinking_delta events during reasoning so the
        # stream doesn't go silent for minutes (Claude 4.7 defaults to "omitted",
        # which emits only signature_delta at the end). Summarized thinking text is
        # not used for token accounting (see _count_tokens_exact), only as heartbeat.
        request_body = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "adaptive", "display": "summarized"},
            "stream": True,
        }

        # output_config.effort: soft guidance for total token spend (Mythos, 4.7,
        # 4.6, Sonnet 4.6, Opus 4.5). Values: low / medium / high / xhigh (4.7 only)
        # / max (not 4.5). `high` is the server default; passing it is equivalent
        # to omitting.
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            request_body["output_config"] = {"effort": reasoning_effort}

        try:
            text = ""
            input_tokens = 0
            output_tokens = 0
            # Cache token fields (informational; not surfaced in TokenUsage schema)
            cache_creation_tokens = 0
            cache_read_tokens = 0
            cache_ephemeral_5m = 0
            cache_ephemeral_1h = 0
            model_name = self.model
            stop_reason = "end_turn"
            buffer = ""

            async with self._http_client.stream(
                "POST", self.messages_url, headers=self.headers, json=request_body
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
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
                                cache_creation_tokens = usage.get("cache_creation_input_tokens", cache_creation_tokens) or cache_creation_tokens
                                cache_read_tokens = usage.get("cache_read_input_tokens", cache_read_tokens) or cache_read_tokens
                                stop_reason = d.get("delta", {}).get("stop_reason", stop_reason)

                        except json.JSONDecodeError:
                            continue

            # Anthropic reports total output_tokens (thinking + answer combined) and
            # never breaks them out in usage. Get exact answer_tokens by running the
            # visible answer text through /v1/messages/count_tokens (real Anthropic
            # tokenizer), then derive reasoning_tokens by subtraction. Works whether
            # thinking.display is "summarized", "omitted", or full plaintext.
            answer_tokens = await self._count_tokens_exact(text) if text else 0
            answer_tokens = min(answer_tokens, output_tokens)
            reasoning_tokens = output_tokens - answer_tokens

            if cache_creation_tokens or cache_read_tokens:
                logger.info(
                    f"cache: creation={cache_creation_tokens} "
                    f"(5m={cache_ephemeral_5m}, 1h={cache_ephemeral_1h}) "
                    f"read={cache_read_tokens}"
                )

            tokens = TokenUsage(
                prompt_tokens=input_tokens,
                answer_tokens=answer_tokens,
                reasoning_tokens=reasoning_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,
                model=model_name,
                finish_reason=stop_reason,
            )

        except httpx.HTTPStatusError as e:
            error_body = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"Anthropic API HTTP error: {e.response.status_code} - {error_body}")
            raise Exception(f"Error code: {e.response.status_code} - {error_body}")
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def _count_tokens_exact(self, text: str) -> int:
        """Return Anthropic's exact token count for `text` via /v1/messages/count_tokens.

        Includes a few tokens of user-message wrapping overhead (~4-6), which is
        noise (<1%) for substantive answers and self-corrects via min(answer, output)
        clamping for very short answers. Falls back to a tiktoken-based estimate
        if the endpoint is unavailable.
        """
        if not text:
            return 0
        try:
            resp = await self._http_client.post(
                self.count_tokens_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": text}],
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return int(resp.json().get("input_tokens", 0))
        except Exception as e:
            logger.warning(
                f"count_tokens failed, falling back to tiktoken estimate: {e}"
            )
            from ..utils.token_counter import estimate_tokens
            return estimate_tokens(text, self.model)
