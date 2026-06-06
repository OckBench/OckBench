"""Google Gemini API client."""
import asyncio
import logging
from typing import Any, Dict, Optional

from google import genai

from ..core.schemas import ModelResponse, TokenUsage
from ..utils.usage_normalizer import extract_gemini_usage, to_token_usage
from .base import BaseModelClient
from .registry import register_provider

logger = logging.getLogger(__name__)


def _retry_disabled_http_options() -> Any:
    """Return http_options that disable the genai SDK's own retry (attempts=1).

    BaseModelClient.generate is the single retry owner; the SDK loop must not
    stack on top of it. Prefers the typed options and falls back to the dict
    form the SDK also accepts, so this works across SDK versions.
    """
    try:
        from google.genai import types
        return types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
    except Exception:  # pragma: no cover - depends on SDK version
        return {"retry_options": {"attempts": 1}}


@register_provider("gemini")
class GeminiClient(BaseModelClient):
    """Client for Google Gemini via the google-genai SDK."""

    protected_paths = ("model", "contents")
    provider_name = "gemini"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        http_options = _retry_disabled_http_options()
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        else:
            self.client = genai.Client(http_options=http_options)

    def build_request(self, prompt: str, max_output_tokens: int) -> Dict[str, Any]:
        config: Dict[str, Any] = {"temperature": self.temperature, "max_output_tokens": max_output_tokens}
        if self.top_p is not None:
            config["top_p"] = self.top_p
        return {"model": self.model, "contents": prompt, "config": config}

    async def _dispatch(self, request: Dict[str, Any]) -> ModelResponse:
        try:
            # config is overridable (nested reasoning controls live under it), so
            # tolerate an override that drops it rather than KeyError-ing.
            response = await self._generate_content_async(
                request["model"], request["contents"], request.get("config") or {}
            )
            text = self._extract_text(response)
            tokens = self._extract_tokens(response)
            finish_reason = self._get_finish_reason(response)

            return ModelResponse(
                text=text, tokens=tokens, latency=0,
                model=request["model"], finish_reason=finish_reason,
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise

    async def _generate_content_async(self, model: str, contents: Any, config: dict):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(model=model, contents=contents, config=config),
        )

    def _extract_text(self, response) -> str:
        if hasattr(response, 'text') and response.text:
            return response.text

        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = getattr(candidate.content, 'parts', None)
                if parts:
                    texts = [p.text for p in parts if hasattr(p, 'text') and p.text]
                    if texts:
                        return ' '.join(texts)
            if hasattr(candidate, 'text') and candidate.text:
                return candidate.text

        tokens = self._extract_tokens(response)
        if tokens.reasoning_tokens > 0:
            logger.warning(f"No text in response, but {tokens.reasoning_tokens} thinking tokens used")
        else:
            logger.warning("No text content found in Gemini response")
        return ""

    def _get_finish_reason(self, response) -> Optional[str]:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                return str(candidate.finish_reason)
        return None

    def _extract_tokens(self, response) -> TokenUsage:
        metadata = getattr(response, 'usage_metadata', None)
        return to_token_usage(extract_gemini_usage(metadata))
