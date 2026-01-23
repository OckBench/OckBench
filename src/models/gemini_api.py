"""
Google Gemini API client implementation.
"""
import logging
from typing import Optional
from google import genai

from .base import BaseModelClient
from ..core.schemas import ModelResponse, TokenUsage


logger = logging.getLogger(__name__)


class GeminiClient(BaseModelClient):
    """
    Client for Google Gemini API.
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
        Initialize Gemini client.
        
        Args:
            model: Model name (e.g., 'gemini-2.5-flash')
            api_key: Google API key
            base_url: Not used for Gemini (included for consistency)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters
        """
        super().__init__(model, api_key, base_url, timeout, max_retries, **kwargs)
        
        # Initialize Gemini client (new SDK)
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
    
    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """Call Gemini API."""
        config = {'temperature': temperature, 'max_output_tokens': max_output_tokens}
        if kwargs.get("top_p") is not None:
            config['top_p'] = kwargs["top_p"]

        try:
            response = await self._generate_content_async(prompt, config)
            text = self._extract_text(response)
            tokens = self._extract_tokens(response)
            finish_reason = self._get_finish_reason(response)

            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,  # Will be set by base class
                model=self.model,
                finish_reason=finish_reason
            )
        
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise
    
    async def _generate_content_async(self, prompt: str, config: dict):
        """
        Wrapper to call synchronous Gemini API in async context.
        
        Args:
            prompt: Input prompt
            config: Generation configuration dictionary
        
        Returns:
            Response from Gemini API
        """
        import asyncio
        
        # Run sync call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
        )
        return response

    def _extract_text(self, response) -> str:
        """Extract text from Gemini response with multiple fallback strategies."""
        # Strategy 1: Direct text attribute
        if hasattr(response, 'text') and response.text:
            return response.text

        # Strategy 2: Extract from candidates structure
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]

            # Try candidate.content.parts
            if hasattr(candidate, 'content') and candidate.content:
                parts = getattr(candidate.content, 'parts', None)
                if parts:
                    texts = [p.text for p in parts if hasattr(p, 'text') and p.text]
                    if texts:
                        return ' '.join(texts)

            # Try direct candidate.text
            if hasattr(candidate, 'text') and candidate.text:
                return candidate.text

        # Log warning if no text extracted
        tokens = self._extract_tokens(response)
        if tokens.reasoning_tokens > 0:
            logger.warning(f"No text in response, but {tokens.reasoning_tokens} thinking tokens used")
        else:
            logger.warning("No text content found in Gemini response")
        return ""

    def _get_finish_reason(self, response) -> Optional[str]:
        """Extract finish reason from Gemini response."""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                return str(candidate.finish_reason)
        return None

    def _extract_tokens(self, response) -> TokenUsage:
        """
        Extract token usage from Gemini response.
        
        Args:
            response: Gemini API response object
        
        Returns:
            TokenUsage: Token usage information
        """
        prompt_tokens = 0
        answer_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0  # For Gemini 2.5 Flash thinking tokens
        
        # Gemini provides usage_metadata
        if hasattr(response, 'usage_metadata'):
            metadata = response.usage_metadata
            
            # Debug: log available attributes (only in debug mode)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Gemini response metadata: {metadata}")
            
            # Try different field names for different SDK versions
            # Use or 0 to handle None values
            prompt_tokens = getattr(metadata, 'prompt_token_count', None) or 0
            answer_tokens = getattr(metadata, 'candidates_token_count', None) or 0
            total_tokens = getattr(metadata, 'total_token_count', None) or 0
            
            # Gemini 2.5 Flash has thinking tokens!
            thoughts_tokens = getattr(metadata, 'thoughts_token_count', None) or 0
            if thoughts_tokens:
                reasoning_tokens = int(thoughts_tokens)
                logger.debug(f"Gemini thinking tokens: {reasoning_tokens}")
            
            # Alternative field names
            if not prompt_tokens:
                prompt_tokens = getattr(metadata, 'prompt_tokens', None) or 0
            if not answer_tokens:
                answer_tokens = getattr(metadata, 'completion_tokens', None) or 0
            if not total_tokens:
                total_tokens = getattr(metadata, 'total_tokens', None) or 0
            
            # New SDK might use input/output terminology
            if not prompt_tokens:
                prompt_tokens = getattr(metadata, 'input_tokens', None) or 0
            if not answer_tokens:
                answer_tokens = getattr(metadata, 'output_tokens', None) or 0
        
        # Ensure we have integers (handle any remaining None values)
        prompt_tokens = int(prompt_tokens) if prompt_tokens else 0
        answer_tokens = int(answer_tokens) if answer_tokens else 0
        total_tokens = int(total_tokens) if total_tokens else 0
        
        # Calculate total if not provided
        # Note: total_tokens includes thinking tokens for Gemini 2.5 Flash
        if not total_tokens:
            total_tokens = prompt_tokens + answer_tokens + reasoning_tokens
        
        # Calculate output_tokens = reasoning_tokens + answer_tokens
        output_tokens = reasoning_tokens + answer_tokens
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            reasoning_tokens=reasoning_tokens,  # Gemini 2.5 Flash has thinking tokens
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )

