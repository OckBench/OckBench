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
    return int(len(text) // 4 * NON_OPENAI_TOKENIZER_MULTIPLIER)

