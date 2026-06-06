"""Run identity: the tuple that makes a cache safe to resume into.

A cache is only safe to resume if the run that produced it is the *same* run in
every way that could change a problem's outcome: same provider/model, same
dataset+split, same prompt/template, same resolved request shape, same
generation settings and output budget, same evaluator/judge identity, and same
scoring/schema version. This module computes that tuple (with secrets excluded)
and a stable hash of it, so resume can compare identities and refuse a silent
merge across different runs.
"""
import hashlib
import json
from typing import Any, Dict

from ..utils.prompt_formatter import template_identity
from ..utils.request_overrides import redact_url
from .schemas import SCHEMA_VERSION, BenchmarkConfig


def compute_run_identity(config: BenchmarkConfig) -> Dict[str, Any]:
    """Return the ordered identity components for ``config`` (no secrets).

    Credentials are intentionally excluded — they do not change a problem's
    outcome and must never be written to a cache header. The endpoint host *is*
    part of identity (the same model at two endpoints is a different run), but
    any credentials embedded in a ``base_url`` are masked first.
    """
    overrides = config.request_overrides
    judge_identity = None
    if config.judge is not None:
        # Judge identity = model + endpoint (never the key); judge request shape
        # affects verdicts, so include it.
        judge_identity = {
            "model": config.judge.model,
            "base_url": redact_url(config.judge.base_url),
            "request_overrides": {
                "set": dict(config.judge.request_overrides.set),
                "unset": list(config.judge.request_overrides.unset),
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "provider": config.provider,
        "model": config.model,
        "base_url": redact_url(config.base_url),
        "dataset": {
            "name": config.dataset_name,
            "split": config.dataset_split,
            "path": config.dataset_path,
        },
        "evaluator_type": config.evaluator_type,
        "prompt_template": template_identity(config.evaluator_type),
        "request_overrides": {
            "set": dict(overrides.set),
            "unset": list(overrides.unset),
        },
        "generation": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.max_output_tokens,
            "max_context_window": config.max_context_window,
        },
        "judge": judge_identity,
    }


def identity_hash(identity: Dict[str, Any]) -> str:
    """Stable SHA-256 over the canonicalized identity components."""
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def diff_identity(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    """Return the top-level identity keys whose values differ, for messaging."""
    changed: Dict[str, Any] = {}
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            changed[key] = {"cache": expected.get(key), "run": actual.get(key)}
    return changed
