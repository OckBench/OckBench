"""
Base class for model API clients.
"""
import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ..core.schemas import ModelResponse, TokenUsage


logger = logging.getLogger(__name__)


class BaseModelClient(ABC):
    """
    Abstract base class for model API clients.
    
    Provides common functionality like retry logic and error handling.
    Subclasses implement provider-specific API calls.
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize base model client.
        
        Args:
            model: Model name/identifier
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional provider-specific parameters
        """
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
        """
        Make API call to model provider.
        
        This method should be implemented by subclasses for specific providers.
        
        Args:
            prompt: Input prompt/question
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            **kwargs: Additional generation parameters
        
        Returns:
            ModelResponse: Response with text and token usage
        """
        pass
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """
        Generate response with retry logic.
        
        Args:
            prompt: Input prompt/question
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            **kwargs: Additional generation parameters
        
        Returns:
            ModelResponse: Response with text and token usage
        """
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
                latency = time.time() - start_time
                response.latency = latency
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Don't retry on certain errors
                if self._is_non_retryable_error(error_msg):
                    logger.error(f"Non-retryable error: {error_msg}")
                    return self._create_error_response(error_msg, time.time() - start_time)
                
                # Log retry attempt
                if attempt < self.max_retries - 1:
                    wait_time = self._get_backoff_time(attempt)
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {error_msg}")
        
        # All retries failed
        return self._create_error_response(str(last_error), 0)
    
    def _get_backoff_time(self, attempt: int) -> float:
        """
        Calculate exponential backoff time.
        
        Args:
            attempt: Current attempt number (0-indexed)
        
        Returns:
            float: Wait time in seconds
        """
        # Exponential backoff: 2^attempt seconds (1s, 2s, 4s, ...)
        return min(2 ** attempt, 30)  # Cap at 30 seconds
    
    def _is_non_retryable_error(self, error_msg: str) -> bool:
        """
        Check if error should not be retried.
        
        Args:
            error_msg: Error message string
        
        Returns:
            bool: True if error should not be retried
        """
        non_retryable_keywords = [
            'invalid api key',
            'authentication failed',
            'invalid model',
            'context length exceeded',
            'content policy violation',
            'invalid request',
            'bad request'
        ]
        
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in non_retryable_keywords)
    
    def _create_error_response(self, error_msg: str, latency: float) -> ModelResponse:
        """
        Create error response when API call fails.
        
        Args:
            error_msg: Error message
            latency: Time taken before error
        
        Returns:
            ModelResponse: Error response
        """
        return ModelResponse(
            text="",
            tokens=TokenUsage(
                prompt_tokens=0,
                answer_tokens=0,
                reasoning_tokens=0,
                output_tokens=0,
                total_tokens=0
            ),
            latency=latency,
            model=self.model,
            error=error_msg,
            finish_reason="error"
        )
    
    def _normalize_model_name(self, full_name: str) -> str:
        """
        Extract short model name from full model identifier.
        
        Args:
            full_name: Full model name/path
        
        Returns:
            str: Normalized model name
        """
        # Extract last part of path-like model names
        if '/' in full_name:
            return full_name.split('/')[-1]
        return full_name

