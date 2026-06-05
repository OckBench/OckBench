"""User-driven request overrides for outgoing chat-completions requests.

This module provides a generic, provider-agnostic mechanism to mutate the
request dict a client assembles, instead of hard-coding per-provider rules.

The override object holds two parts:
  - ``set``:   a mapping of dotted-path -> JSON-typed value (deep create/replace)
  - ``unset``: a list of dotted paths to remove (deep delete)

Values in ``set`` may reference dynamic, per-request quantities through
``${name}`` placeholders (currently ``${max_output_tokens}``), substituted at
request-build time. Structural fields a client depends on are protected by a
prefix-based guard that must be run once on the merged override object before
application (see ``guard_protected_paths``).
"""
import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# Fields the chat-completions client depends on for streaming and token
# accounting. A protected path blocks itself and anything nested beneath it.
CHAT_COMPLETION_PROTECTED_PATHS = ("model", "messages", "stream", "stream_options")

# Leaf keys whose values are masked when persisting overrides to saved output.
SECRET_KEY_PATTERNS = ("key", "token", "secret", "authorization", "password")

MASK = "***MASKED***"


def _override_part(overrides: Any, attr: str, default):
    """Read ``set``/``unset`` from either a model object or a plain mapping."""
    if overrides is None:
        return default
    if isinstance(overrides, Mapping):
        value = overrides.get(attr, default)
    else:
        value = getattr(overrides, attr, default)
    return default if value is None else value


def substitute_dynamic_values(value: Any, dynamic: Mapping[str, Any]) -> Any:
    """Substitute ``${name}`` placeholders inside an override value.

    An exact ``"${name}"`` string yields the dynamic value with its original
    type (e.g. an int budget). A placeholder embedded in a larger string is
    replaced textually. Dicts and lists are handled recursively.
    """
    if isinstance(value, str):
        for name, dyn_value in dynamic.items():
            if value == "${" + name + "}":
                return dyn_value
        result = value
        for name, dyn_value in dynamic.items():
            token = "${" + name + "}"
            if token in result:
                result = result.replace(token, str(dyn_value))
        return result
    if isinstance(value, dict):
        return {key: substitute_dynamic_values(val, dynamic) for key, val in value.items()}
    if isinstance(value, list):
        return [substitute_dynamic_values(item, dynamic) for item in value]
    return value


def _set_by_path(target: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _unset_by_path(target: Dict[str, Any], path: str) -> None:
    keys = path.split(".")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            return
        node = child
    node.pop(keys[-1], None)


def guard_protected_paths(paths: Iterable[str], protected_paths: Sequence[str]) -> None:
    """Reject any path that targets a protected field (prefix semantics).

    A protected path ``P`` blocks ``P`` and any path prefixed by ``P.``.
    Raises ``ValueError`` naming the first offending path.
    """
    for path in paths:
        for protected in protected_paths:
            if path == protected or path.startswith(protected + "."):
                raise ValueError(
                    f"request override path '{path}' targets protected field "
                    f"'{protected}', which OckBench manages and cannot be overridden"
                )


def apply_request_overrides(
    request: Dict[str, Any],
    overrides: Any,
    dynamic: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply ``set`` (deep create/replace) then ``unset`` (deep delete).

    ``overrides`` may be a ``RequestOverrides`` model or a plain mapping with
    ``set``/``unset`` keys. The protected-field guard is intentionally NOT run
    here; it runs once on the merged object during configuration assembly.
    """
    set_map: Dict[str, Any] = _override_part(overrides, "set", {})
    unset_list: List[str] = _override_part(overrides, "unset", [])
    dynamic = dynamic or {}

    for path, value in set_map.items():
        _set_by_path(request, path, substitute_dynamic_values(value, dynamic))
    for path in unset_list:
        _unset_by_path(request, path)
    return request


def _is_secret_key(key: str) -> bool:
    lowered = str(key).lower()
    for pattern in SECRET_KEY_PATTERNS:
        if pattern == "token":
            # Match credential tokens (singular "token", e.g. access_token) but
            # not token-budget fields like max_tokens / max_completion_tokens,
            # whose values are needed to reproduce a run.
            if "token" in lowered and "tokens" not in lowered:
                return True
        elif pattern in lowered:
            return True
    return False


def _redact_value(value: Any) -> Any:
    """Recursively mask secret-keyed entries inside an override value.

    Override values may be nested objects (from ``--request-set-json`` or YAML),
    so a secret can hide below the top-level dotted path, e.g.
    ``extra_headers={"Authorization": "Bearer ..."}``.
    """
    if isinstance(value, dict):
        return {
            key: (MASK if _is_secret_key(key) else _redact_value(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def redact_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a serialized config with secrets masked.

    Masks the top-level ``api_key`` and, for every ``request_overrides.set``
    entry, masks the whole value when the leaf path key is secret-like and
    otherwise recurses into the value to mask any nested secret-keyed entries.
    """
    redacted = copy.deepcopy(config_dict)

    if redacted.get("api_key"):
        redacted["api_key"] = MASK

    overrides = redacted.get("request_overrides")
    if isinstance(overrides, dict):
        set_map = overrides.get("set")
        if isinstance(set_map, dict):
            overrides["set"] = {
                path: (MASK if _is_secret_key(path.split(".")[-1]) else _redact_value(value))
                for path, value in set_map.items()
            }

    return redacted
