"""Google Gemini API client."""
import asyncio
import logging
from typing import Optional

from google import genai

from ..core.schemas import ModelResponse, TokenUsage
from ..utils.usage_normalizer import extract_gemini_usage, to_token_usage
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class GeminiClient(BaseModelClient):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: int = 120, max_retries: int = 3, **kwargs):
        super().__init__(model, api_key, base_url, timeout, max_retries, **kwargs)
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    async def _call_api(self, prompt: str, temperature: float = 0.0,
                        max_output_tokens: int = 4096, **kwargs) -> ModelResponse:
        config = {'temperature': temperature, 'max_output_tokens': max_output_tokens}
        if kwargs.get("top_p") is not None:
            config['top_p'] = kwargs["top_p"]

        try:
            response = await self._generate_content_async(prompt, config)
            text = self._extract_text(response)
            tokens = self._extract_tokens(response)
            finish_reason = self._get_finish_reason(response)

            return ModelResponse(
                text=text, tokens=tokens, latency=0,
                model=self.model, finish_reason=finish_reason,
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise

    async def _generate_content_async(self, prompt: str, config: dict):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model, contents=prompt, config=config,
            ),
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
