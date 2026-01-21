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

    For non-OpenAI models, applies a multiplier to account for tokenizer differences.
    Different tokenizers can produce significantly different token counts for the
    same text (e.g., Qwen3 tokenizer vs GPT-4 tokenizer can differ by 50-70%).

    Args:
        text: Input text
        model: Model name for encoding

    Returns:
        int: Estimated token count
    """
    # Multiplier for non-OpenAI models to account for tokenizer differences
    # Based on empirical observation: Qwen3 tokenizer produces ~1.65x more tokens
    # than tiktoken's cl100k_base for the same text. Using 1.7 for safety margin.
    NON_OPENAI_TOKENIZER_MULTIPLIER = 1.7

    if TIKTOKEN_AVAILABLE:
        try:
            # Try to get encoding for specific model (works for OpenAI models)
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            # Unknown model - use cl100k_base with multiplier
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                base_count = len(encoding.encode(text))
                # Apply multiplier for non-OpenAI models
                adjusted_count = int(base_count * NON_OPENAI_TOKENIZER_MULTIPLIER)
                logger.debug(
                    f"Non-OpenAI model '{model}': base tokens={base_count}, "
                    f"adjusted tokens={adjusted_count} (multiplier={NON_OPENAI_TOKENIZER_MULTIPLIER})"
                )
                return adjusted_count
            except Exception:
                pass

    # Fallback: rough estimation (4 characters ≈ 1 token for English)
    # Also apply multiplier for safety
    return int(len(text) // 4 * NON_OPENAI_TOKENIZER_MULTIPLIER)


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

