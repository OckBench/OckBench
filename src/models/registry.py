"""Provider registry: the single seam for selecting and constructing clients.

A provider name maps to a client factory (a ``BaseModelClient`` subclass)
through one registry. The runner and the inspect surface both resolve providers
here, so there is no per-provider ``if/elif`` construction chain anywhere. The
four built-in providers register themselves on import (see ``src/models``);
external users register their own client the same way — by importing a module
that calls ``@register_provider("name")`` — without editing the runner, the
config schema, or this file.
"""
from typing import Callable, Dict, List, Type

# name -> client factory. Populated by @register_provider at import time.
_PROVIDER_REGISTRY: Dict[str, Callable[..., "object"]] = {}


def register_provider(name: str) -> Callable[[Type], Type]:
    """Class decorator that registers a client factory under ``name``.

    Re-registering an existing name is rejected so an accidental shadow of a
    built-in fails loudly rather than silently winning.
    """
    def _decorator(factory: Type) -> Type:
        if name in _PROVIDER_REGISTRY:
            raise ValueError(f"provider '{name}' is already registered")
        _PROVIDER_REGISTRY[name] = factory
        return factory

    return _decorator


def available_providers() -> List[str]:
    """Return the registered provider names, sorted for stable messages."""
    return sorted(_PROVIDER_REGISTRY)


def create_provider(name: str, **kwargs):
    """Construct the client registered under ``name``.

    Raises ``ValueError`` enumerating the available providers when ``name`` is
    not registered (fail fast, no silent default).
    """
    factory = _PROVIDER_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"unknown provider '{name}'. Registered providers: "
            f"{', '.join(available_providers()) or '(none)'}"
        )
    return factory(**kwargs)
