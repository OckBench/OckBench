"""Cross-provider dry-run / inspect surface (no network).

``build_inspection`` resolves a config into the exact sanitized request a run
would send, for *any* provider, using the same ``shape_request`` code path the
runner uses — but without opening a socket. Every secret (api_key, bearer
tokens, credentialed base_url, secret headers, and the math judge's
credentials) is masked, so inspect output is safe to paste into a bug report.
"""
import json
from typing import Any, Dict

from ..models import create_provider
from ..utils.prompt_formatter import format_prompt
from ..utils.request_overrides import MASK, redact_request, redact_url
from .schemas import BenchmarkConfig

# A representative problem so the resolved request is concrete and readable.
_SAMPLE_PROBLEM = "What is 2 + 2?"


def _inspect_budget(config: BenchmarkConfig, prompt: str) -> int:
    """The max-output-tokens value a run would use for this prompt."""
    if config.max_output_tokens is not None:
        return config.max_output_tokens
    # max_context_window mode: mirror the runner's dynamic budget calc.
    from ..utils.token_counter import estimate_tokens
    try:
        input_tokens = estimate_tokens(prompt, config.model)
    except Exception:
        input_tokens = len(prompt) // 4
    return max(config.max_context_window - input_tokens - 256, 100)


def build_inspection(config: BenchmarkConfig) -> Dict[str, Any]:
    """Resolve ``config`` into a sanitized inspection dict (opens no socket)."""
    client = create_provider(
        config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
        temperature=config.temperature,
        top_p=config.top_p,
        request_overrides=config.request_overrides,
    )

    test_cases = ["assert solution(1) == 1"] if config.evaluator_type == "code" else None
    sample_prompt = format_prompt(
        problem=_SAMPLE_PROBLEM, evaluator_type=config.evaluator_type, test_cases=test_cases,
    )
    max_output_tokens = _inspect_budget(config, sample_prompt)
    request = client.shape_request(sample_prompt, max_output_tokens)

    inspection: Dict[str, Any] = {
        "provider": config.provider,
        "model": config.model,
        "base_url": redact_url(config.base_url) if config.base_url else None,
        "api_key": MASK if config.api_key else None,
        "evaluator_type": config.evaluator_type,
        "max_output_tokens": max_output_tokens,
        "request": redact_request(request),
    }

    if config.judge is not None:
        inspection["judge"] = {
            "model": config.judge.model,
            "base_url": redact_url(config.judge.base_url) if config.judge.base_url else None,
            "api_key": MASK if config.judge.api_key else None,
        }

    return inspection


def format_inspection(inspection: Dict[str, Any]) -> str:
    return json.dumps(inspection, indent=2, ensure_ascii=False, default=str)
