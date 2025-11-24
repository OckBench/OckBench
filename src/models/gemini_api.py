"""
Google Gemini API client implementation.
"""
import logging
from typing import Optional
from google import genai

from .base import BaseModelClient
from ..core.schemas import ModelResponse, TokenUsage
from ..utils.prompt_formatter import format_prompt


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
        """
        Call Gemini API (new SDK).
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            **kwargs: Additional parameters (top_p, enforce_format, etc.)
        
        Returns:
            ModelResponse: Response with text and tokens
        """
        # Extract format enforcement params
        enforce_format = kwargs.pop('enforce_output_format', False)
        custom_instruction = kwargs.pop('custom_format_instruction', None)
        evaluator_type = kwargs.pop('evaluator_type', 'math')
        
        # Format prompt with optional instruction
        formatted_prompt = format_prompt(
            problem=prompt,
            enforce_format=enforce_format,
            custom_instruction=custom_instruction,
            evaluator_type=evaluator_type
        )
        
        # Build generation config for new SDK
        config = {
            'temperature': temperature,
            'max_output_tokens': max_output_tokens,
        }
        
        # Add optional parameters
        if kwargs.get("top_p") is not None:
            config['top_p'] = kwargs["top_p"]
        
        try:
            # Make API call using new SDK (synchronous call in async context)
            # The new SDK doesn't have native async support yet
            response = await self._generate_content_async(formatted_prompt, config)
            
            # Debug: log response structure
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Gemini response type: {type(response)}")
                logger.debug(f"Gemini response has text attr: {hasattr(response, 'text')}")
                logger.debug(f"Gemini response has candidates: {hasattr(response, 'candidates')}")
            
            # Extract response text with multiple fallback strategies
            text = ""
            
            # Strategy 1: Direct text attribute
            if hasattr(response, 'text') and response.text:
                text = response.text
            
            # Strategy 2: Extract from candidates structure
            elif hasattr(response, 'candidates') and response.candidates:
                if len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    
                    # Try candidate.content.parts
                    if hasattr(candidate, 'content') and candidate.content:
                        content = candidate.content
                        if hasattr(content, 'parts') and content.parts:
                            try:
                                parts_texts = []
                                for part in content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        parts_texts.append(part.text)
                                if parts_texts:
                                    text = ' '.join(parts_texts)
                            except (TypeError, AttributeError) as e:
                                logger.warning(f"Failed to extract from parts: {e}")
                    
                    # Try direct candidate.text
                    if not text and hasattr(candidate, 'text') and candidate.text:
                        text = candidate.text
            
            # Log warning if no text extracted
            if not text:
                # Check if model did thinking but produced no output
                tokens_temp = self._extract_tokens(response)
                if tokens_temp.reasoning_tokens > 0:
                    logger.warning(
                        f"No text content found in Gemini response, "
                        f"but model used {tokens_temp.reasoning_tokens} thinking tokens. "
                        f"Model may have decided not to answer after internal reasoning or the reasoning exceeded the max output tokens."
                    )
                else:
                    logger.warning("No text content found in Gemini response")
            
            # Extract token usage
            tokens = self._extract_tokens(response)
            
            # Get finish reason if available
            finish_reason = None
            if hasattr(response, 'candidates') and response.candidates:
                if len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason)
            
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

