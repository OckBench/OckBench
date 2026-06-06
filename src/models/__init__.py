"""API clients for different model providers.

Importing this package registers the four built-in providers with the provider
registry (each client module calls ``@register_provider`` at import time), so
``create_provider(name)`` can resolve them. External providers register the same
way by importing their own module.
"""
from .anthropic_api import AnthropicClient
from .gemini_api import GeminiClient
from .openai_api import OpenAIClient
from .openai_responses_api import OpenAIResponsesClient
from .registry import available_providers, create_provider, register_provider

__all__ = [
    "AnthropicClient",
    "GeminiClient",
    "OpenAIClient",
    "OpenAIResponsesClient",
    "available_providers",
    "create_provider",
    "register_provider",
]
