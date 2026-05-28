"""Base class for model API clients."""
import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..core.schemas import ModelResponse, TokenUsage

logger = logging.getLogger(__name__)


class BaseModelClient(ABC):
    """Abstract base class for model API clients with retry logic."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_params = kwargs

    @abstractmethod
    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        pass

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """Generate response with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = await self._call_api(
                    prompt=prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    **kwargs
                )
                response.latency = time.time() - start_time
                return response

            except Exception as e:
                last_error = e
                error_msg = str(e)

                if self._is_non_retryable_error(e):
                    logger.error(f"Non-retryable error: {error_msg}")
                    return self._create_error_response(error_msg, time.time() - start_time)

                if attempt < self.max_retries - 1:
                    wait_time = self._retry_wait_time(error_msg, attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {error_msg}")

        return self._create_error_response(str(last_error), 0)

    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Check if error should not be retried."""
        status = getattr(error, 'status_code', None)
        if status in (400, 401, 403, 404, 422):
            return True

        non_retryable_keywords = [
            'invalid api key',
            'authentication failed',
            'invalid model',
            'context length exceeded',
            'content policy violation',
            'invalid request',
        ]
        error_lower = str(error).lower()
        return any(keyword in error_lower for keyword in non_retryable_keywords)

    def _retry_wait_time(self, error_msg: str, attempt: int) -> float:
        """Back off longer for provider throttling and broken streaming bodies."""
        error_lower = error_msg.lower()
        retry_pressure_keywords = [
            "429",
            "rate limit",
            "too many requests",
            "peer closed",
            "incomplete message body",
            "remoteprotocolerror",
            "readtimeout",
            "timeout",
        ]
        if any(keyword in error_lower for keyword in retry_pressure_keywords):
            base_wait = min(10 * (2 ** attempt), 120)
        else:
            base_wait = min(2 ** attempt, 30)
        return round(base_wait + random.uniform(0, 1.5), 2)

    def _create_error_response(self, error_msg: str, latency: float) -> ModelResponse:
        return ModelResponse(
            text="",
            tokens=TokenUsage(
                prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                output_tokens=0, total_tokens=0,
            ),
            latency=latency,
            model=self.model,
            error=error_msg,
            finish_reason="error",
        )
