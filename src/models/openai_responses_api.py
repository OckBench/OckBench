"""OpenAI Responses API client (/v1/responses) with streaming."""
import json
import logging
from typing import Optional

import httpx

from ..core.schemas import ModelResponse
from ..utils.usage_normalizer import extract_responses_usage, to_token_usage
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class OpenAIResponsesClient(BaseModelClient):
    """Client for OpenAI /v1/responses (models that only support this endpoint)."""

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

        self.endpoint = (base_url or "https://api.openai.com/v1").rstrip("/")
        if not self.endpoint.endswith("/v1"):
            if "/v1" not in self.endpoint:
                self.endpoint += "/v1"
        self.responses_url = self.endpoint + "/responses"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'dummy-key'}",
        }

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        request_body = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "stream": True,
        }

        # Reasoning models take reasoning.effort and reject temperature; others
        # take temperature. Set via the reasoning_effort config field (YAML).
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort:
            request_body["reasoning"] = {"effort": reasoning_effort}
        else:
            request_body["temperature"] = temperature

        try:
            text = ""
            model_name = self.model
            usage_payload = {}
            status = "completed"
            buffer = ""
            saw_done = False
            completed_event = False

            async with self._http_client.stream(
                "POST", self.responses_url, headers=self.headers, json=request_body
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
                            saw_done = True
                            break
                        try:
                            d = json.loads(data_str)
                            event_type = d.get("type")

                            if event_type == "response.output_text.delta":
                                text += d.get("delta", "")

                            elif event_type == "response.completed":
                                completed_event = True
                                resp_data = d.get("response", {})
                                model_name = resp_data.get("model", model_name)
                                status = resp_data.get("status", status)
                                usage_payload = resp_data.get("usage", {})

                        except json.JSONDecodeError:
                            continue
                    if saw_done:
                        break

            normalized = extract_responses_usage(usage_payload)
            tokens = to_token_usage(normalized)
            reasoning_tokens = normalized.reasoning_tokens

            error = None
            if not saw_done:
                error = "responses_stream_incomplete: stream ended before [DONE]"
            elif not completed_event:
                error = "responses_stream_incomplete: no response.completed event"
            elif not text:
                if reasoning_tokens > 0:
                    error = (
                        "empty_response_reasoning_only: responses stream completed with "
                        f"reasoning tokens but no output text (status={status})"
                    )
                else:
                    error = (
                        "empty_response_no_content: responses stream completed with no "
                        f"output text (status={status})"
                    )

            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,
                model=model_name,
                finish_reason=status,
                error=error,
            )

        except httpx.HTTPStatusError as e:
            error_body = e.response.text if hasattr(e.response, 'text') else str(e)
            logger.error(f"Responses API HTTP error: {e.response.status_code} - {error_body}")
            raise Exception(f"Error code: {e.response.status_code} - {error_body}")
        except Exception as e:
            logger.error(f"Responses API error: {e}")
            raise
