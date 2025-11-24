"""
OpenAI API client implementation.

Supports:
- Official OpenAI API
- OpenAI-compatible APIs (vLLM, SGLang, local serving)
- O1/O3 models with reasoning tokens
"""
import logging
from typing import Optional
from openai import AsyncOpenAI

from .base import BaseModelClient
from ..core.schemas import ModelResponse, TokenUsage
from ..utils.prompt_formatter import format_prompt


logger = logging.getLogger(__name__)


class OpenAIClient(BaseModelClient):
    """
    Client for OpenAI API and OpenAI-compatible endpoints.
    
    Supports standard models (GPT-4, GPT-3.5) and reasoning models (O1, O3).
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
        Initialize OpenAI client.
        
        Args:
            model: Model name (e.g., 'gpt-5', 'o3')
            api_key: OpenAI API key
            base_url: Base URL for API (for compatible endpoints)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters
        """
        super().__init__(model, api_key, base_url, timeout, max_retries, **kwargs)
        
        # Initialize OpenAI client
        client_kwargs = {
            'api_key': api_key or 'dummy-key',  # Some local servers don't need real key
            'timeout': timeout,
        }
        
        if base_url:
            client_kwargs['base_url'] = base_url
        
        self.client = AsyncOpenAI(**client_kwargs)
    
    async def _call_api(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """
        Call OpenAI API.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            **kwargs: Additional parameters (top_p, reasoning_effort, enforce_format, etc.)
        
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
        
        # Build messages
        messages = [
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Build request parameters
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Handle reasoning models (gpt-5, o1, o3) vs standard models
        if self._is_reasoning_model():
            # Reasoning models use max_completion_tokens and reasoning_effort
            request_params["max_completion_tokens"] = max_output_tokens
            
            if kwargs.get("reasoning_effort"):
                request_params["reasoning_effort"] = kwargs["reasoning_effort"]
            
            # Reasoning models (gpt-5, o1, o3) don't support temperature, top_p, top_k, 
            # repeat_penalty and other sampling parameters
            # Remove these parameters for all reasoning models
            request_params.pop("temperature", None)
            request_params.pop("top_p", None)
            request_params.pop("top_k", None)
            request_params.pop("repeat_penalty", None)
            # Remove any other sampling-related parameters that might be in kwargs
            for param in ["top_p", "top_k", "repeat_penalty", "frequency_penalty", "presence_penalty"]:
                kwargs.pop(param, None)
        else:
            # Standard models use max_tokens
            request_params["max_tokens"] = max_output_tokens
            
            # Add optional parameters for standard models
            if kwargs.get("top_p") is not None:
                request_params["top_p"] = kwargs["top_p"]
            if kwargs.get("top_k") is not None:
                request_params["top_k"] = kwargs["top_k"]
            if kwargs.get("frequency_penalty") is not None:
                request_params["frequency_penalty"] = kwargs["frequency_penalty"]
            if kwargs.get("presence_penalty") is not None:
                request_params["presence_penalty"] = kwargs["presence_penalty"]
        
        # Make API call
        try:
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response text
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            # Extract token usage
            tokens = self._extract_tokens(response)
            
            return ModelResponse(
                text=text,
                tokens=tokens,
                latency=0,  # Will be set by base class
                model=response.model,
                finish_reason=finish_reason
            )
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _extract_tokens(self, response) -> TokenUsage:
        """
        Extract token usage from OpenAI response.
        
        Args:
            response: OpenAI API response object
        
        Returns:
            TokenUsage: Token usage information
        """
        usage = response.usage
        
        # Standard token fields
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        
        # Reasoning tokens (for o1/o3/gpt-5 models)
        # These models may have completion_tokens_details with reasoning_tokens
        reasoning_tokens = 0
        if hasattr(usage, 'completion_tokens_details'):
            details = usage.completion_tokens_details
            if details and hasattr(details, 'reasoning_tokens'):
                reasoning_tokens = details.reasoning_tokens or 0
        
        # For reasoning models, completion_tokens is the TOTAL output (includes reasoning)
        # So answer_tokens = completion_tokens - reasoning_tokens
        # For non-reasoning models, reasoning_tokens is 0, so answer_tokens = completion_tokens
        answer_tokens = completion_tokens - reasoning_tokens
        
        # output_tokens = reasoning_tokens + answer_tokens = completion_tokens
        output_tokens = completion_tokens
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            total_tokens=getattr(usage, 'total_tokens', 0)
        )
    
    def _is_reasoning_model(self) -> bool:
        """
        Check if model is a reasoning model (o1, o3, gpt-5).
        
        Returns:
            bool: True if reasoning model
        """
        model_lower = self.model.lower()
        return any(prefix in model_lower for prefix in ['o1', 'o3', 'gpt-5'])

