"""OpenAI Responses API client (/v1/responses) with streaming."""
import json
import logging
from typing import Any, Dict

import httpx

from ..core.schemas import ModelResponse
from ..utils.usage_normalizer import extract_responses_usage, to_token_usage
from .base import BaseModelClient, raise_status_error
from .registry import register_provider

logger = logging.getLogger(__name__)


@register_provider("openai-responses")
class OpenAIResponsesClient(BaseModelClient):
    """Client for OpenAI /v1/responses (models that only support this endpoint).

    Reasoning placement is not hard-coded: by default the base request carries
    ``temperature``; a reasoning model is configured by overriding the request
    (e.g. set ``reasoning.effort`` and unset ``temperature``), uniformly with
    every other provider.
    """

    protected_paths = ("model", "input", "stream")
    provider_name = "openai-responses"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.endpoint = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        if "/v1" not in self.endpoint:
            self.endpoint += "/v1"
        self.responses_url = self.endpoint + "/responses"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'dummy-key'}",
        }

        # httpx does not auto-retry; BaseModelClient.generate is the sole retry owner.
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=30.0),
        )

    async def aclose(self) -> None:
        await self._http_client.aclose()

    def build_request(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "stream": True,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        return request

    #: Terminal SSE events (authoritative end-of-stream signals) and the status
    #: each implies when the payload omits an explicit ``response.status``.
    _TERMINAL_EVENTS = {
        "response.completed": "completed",
        "response.incomplete": "incomplete",
        "response.failed": "failed",
    }

    async def _dispatch(self, request: Dict[str, Any]) -> ModelResponse:
        try:
            text = ""
            reasoning_chars = 0
            model_name = self.model
            usage_payload = {}
            status = None
            incomplete_reason = None
            failed_message = None
            buffer = ""
            ended = False

            async with self._http_client.stream(
                "POST", self.responses_url, headers=self.headers, json=request
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
                            # Optional transport sentinel (a Chat Completions
                            # convention some proxies append); the terminal
                            # events below are the authoritative end signal.
                            ended = True
                            break
                        try:
                            d = json.loads(data_str)
                            event_type = d.get("type")

                            if event_type == "response.output_text.delta":
                                text += d.get("delta", "")

                            elif event_type == "response.reasoning_text.delta":
                                reasoning_chars += len(d.get("delta", "") or "")

                            elif event_type in self._TERMINAL_EVENTS:
                                resp_data = d.get("response", {})
                                model_name = resp_data.get("model", model_name)
                                status = resp_data.get("status") or self._TERMINAL_EVENTS[event_type]
                                usage_payload = resp_data.get("usage") or usage_payload
                                details = resp_data.get("incomplete_details") or {}
                                incomplete_reason = details.get("reason")
                                err = resp_data.get("error")
                                if isinstance(err, dict):
                                    failed_message = err.get("message")
                                elif err:
                                    failed_message = str(err)
                                ended = True
                                break

                        except json.JSONDecodeError:
                            continue
                    if ended:
                        break

            normalized = extract_responses_usage(usage_payload, final_text=text)
            tokens = to_token_usage(normalized)
            reasoning_tokens = normalized.reasoning_tokens

            error = None
            if status is None:
                error = "responses_stream_incomplete: stream ended without a terminal event"
            elif status == "failed":
                error = f"responses_stream_failed: {failed_message or 'provider reported failure'}"
            elif status == "incomplete":
                if incomplete_reason == "max_output_tokens" and tokens.output_tokens > 0:
                    # Budget exhaustion is a real, scoreable outcome (same
                    # semantics as anthropic max_tokens): keep usage, no retry.
                    error = None
                else:
                    error = (
                        f"responses_incomplete: reason={incomplete_reason or 'unknown'} "
                        f"(output_tokens={tokens.output_tokens})"
                    )
            elif not text:
                if reasoning_tokens > 0 or reasoning_chars > 0:
                    error = (
                        "empty_response_reasoning_only: responses stream completed with "
                        f"reasoning but no output text (status={status})"
                    )
                else:
                    error = (
                        "empty_response_no_content: responses stream completed with no "
                        f"output text (status={status})"
                    )

            # Keep the incomplete reason auditable on scoreable rows too.
            finish_reason = status
            if status == "incomplete" and incomplete_reason:
                finish_reason = f"incomplete:{incomplete_reason}"

            return ModelResponse(
                text=text, tokens=tokens, latency=0, model=model_name,
                finish_reason=finish_reason, error=error,
            )

        except httpx.HTTPStatusError as e:
            error_body = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"Responses API HTTP error: {e.response.status_code} - {error_body}")
            # Carry status_code so the base layer can classify non-retryable errors.
            raise_status_error(e.response.status_code, error_body)
        except Exception as e:
            logger.error(f"Responses API error: {e}")
            raise
