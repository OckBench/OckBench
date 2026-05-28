"""Token counting utilities for pre-request estimation."""
import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Non-OpenAI tokenizers produce ~1.65x more tokens than cl100k_base (empirical).
NON_OPENAI_TOKENIZER_MULTIPLIER = 1.7


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count. Uses tiktoken if available, else ~4 chars/token."""
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return int(len(encoding.encode(text)) * NON_OPENAI_TOKENIZER_MULTIPLIER)
            except Exception:
                pass

    return int(len(text) // 4 * NON_OPENAI_TOKENIZER_MULTIPLIER)
