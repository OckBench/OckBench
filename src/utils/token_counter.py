"""
Token counting utilities.

Note: Token counting is currently handled by API providers directly.
This module provides pre-request token estimation for max_context_window feature.
"""
import logging

logger = logging.getLogger(__name__)

# Try to import tiktoken (optional dependency)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.debug("tiktoken not available, using rough estimation")


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Estimate token count for text.
    
    If tiktoken is available, uses accurate tokenization.
    Otherwise, uses rough character-based estimation (~4 chars per token).
    
    Args:
        text: Input text
        model: Model name for encoding
    
    Returns:
        int: Estimated token count
    """
    if TIKTOKEN_AVAILABLE:
        try:
            # Try to get encoding for specific model
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            # Fallback to cl100k_base for unknown models
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass
    
    # Fallback: rough estimation (4 characters ≈ 1 token for English)
    return len(text) // 4


def check_context_limit(prompt: str, max_output: int, model: str = "gpt-4") -> bool:
    """
    Check if request fits within context window.
    
    Args:
        prompt: Input prompt
        max_output: Maximum output tokens
        model: Model name
    
    Returns:
        bool: True if within limits
    """
    context_limits = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 16385,
        "o1-preview": 128000,
        "o1-mini": 128000,
    }
    
    limit = context_limits.get(model, 8192)
    prompt_tokens = estimate_tokens(prompt, model)
    
    return (prompt_tokens + max_output) <= limit

