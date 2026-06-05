"""Pydantic schemas for OckBench."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..utils.request_overrides import CHAT_COMPLETION_PROTECTED_PATHS, guard_protected_paths

logger = logging.getLogger(__name__)


class RequestOverrides(BaseModel):
    """User-driven overrides applied to the outgoing request.

    ``set`` maps a dotted path to a JSON-typed value (deep create/replace);
    ``unset`` lists dotted paths to remove (deep delete). Applied in that order.
    """

    set: Dict[str, Any] = Field(default_factory=dict)
    unset: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_paths(self) -> 'RequestOverrides':
        # Pydantic has already coerced/validated every path to str by this
        # after-validator, so only the empty/whitespace case remains to check.
        for path in list(self.set.keys()) + self.unset:
            if not path.strip():
                raise ValueError("request override paths must be non-empty strings")
        return self


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark experiment."""

    dataset_path: str = Field(...)
    dataset_name: Optional[str] = Field(None)

    provider: Literal["chat_completion", "openai-responses", "anthropic", "gemini"] = Field(...)
    model: str = Field(...)
    base_url: Optional[str] = Field(None)
    api_key: Optional[str] = Field(None)

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(None, gt=0)
    max_context_window: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Retained for the openai-responses/anthropic providers, which apply it
    # directly. For chat_completion it is removed (use request_overrides) and
    # rejected by _handle_removed_fields. There is no CLI flag for it anymore.
    reasoning_effort: Optional[str] = Field(None)

    request_overrides: RequestOverrides = Field(default_factory=RequestOverrides)

    concurrency: int = Field(5, gt=0)
    timeout: int = Field(120, gt=0)
    max_retries: int = Field(3, gt=0)

    evaluator_type: str = Field("math")
    execution_timeout: int = Field(5, gt=0)
    include_challenge_tests: bool = Field(True)

    experiment_name: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)

    @model_validator(mode='before')
    @classmethod
    def _handle_removed_fields(cls, data: Any) -> Any:
        # For chat_completion, provider/model placement is now user-driven, so
        # the old reasoning_effort / enable_thinking inputs are removed there:
        # reject a non-null value loudly with a request_overrides migration hint.
        # (A null leftover from an older serialized config is not an intentional
        # setting and is simply ignored.)
        #
        # reasoning_effort remains a working field for the openai-responses and
        # anthropic providers (applied directly by those clients). enable_thinking
        # was a chat_completion-only concern and stays fully removed; for the
        # other providers it is ignored with a warning.
        if not isinstance(data, dict):
            return data
        provider = data.get('provider')

        if provider == 'chat_completion':
            rejected = {
                'reasoning_effort': "set 'reasoning_effort' via request_overrides",
                'enable_thinking': "set 'extra_body.chat_template_kwargs.enable_thinking' via request_overrides",
            }
            present = [key for key in rejected if data.get(key) is not None]
            if present:
                hints = "; ".join(f"{key} -> {rejected[key]}" for key in present)
                raise ValueError(
                    f"removed config field(s) {present} for chat_completion: configure these "
                    f"through request_overrides instead (see the README migration table). {hints}"
                )
        elif data.get('enable_thinking') is not None:
            logger.warning(
                "Ignoring removed config field 'enable_thinking' for provider '%s'; it was a "
                "chat_completion-only setting.", provider,
            )
        return data

    @model_validator(mode='after')
    def validate_config(self) -> 'BenchmarkConfig':
        if self.provider == 'chat_completion':
            if not self.api_key:
                raise ValueError(
                    "chat_completion provider requires --api-key"
                )
            if not self.base_url:
                raise ValueError(
                    "chat_completion provider requires --base-url"
                )

        has_max_output = self.max_output_tokens is not None
        has_max_context = self.max_context_window is not None

        if has_max_output and has_max_context:
            raise ValueError(
                "max_output_tokens and max_context_window are mutually exclusive."
            )
        if not has_max_output and not has_max_context:
            raise ValueError(
                "Either max_output_tokens or max_context_window must be set."
            )

        # Single chokepoint for the protected-field guard: runs once here on the
        # merged override object, so every config-construction path (CLI and
        # YAML/config_path) is covered, not just the CLI parser.
        if self.provider == 'chat_completion':
            guard_protected_paths(
                list(self.request_overrides.set.keys()) + self.request_overrides.unset,
                CHAT_COMPLETION_PROTECTED_PATHS,
            )
        return self


class Problem(BaseModel):
    problem: str = Field(...)
    answer: Any = Field(...)
    id: Any = Field(...)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(0)
    answer_tokens: int = Field(0)
    reasoning_tokens: int = Field(0)
    output_tokens: int = Field(0)
    total_tokens: int = Field(0)

    def __init__(self, **data):
        if 'completion_tokens' in data and 'answer_tokens' not in data:
            data['answer_tokens'] = data.pop('completion_tokens')
        super().__init__(**data)
        if self.output_tokens == 0:
            self.output_tokens = self.reasoning_tokens + self.answer_tokens
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.answer_tokens + self.reasoning_tokens


class ModelResponse(BaseModel):
    text: str = Field(...)
    tokens: TokenUsage = Field(...)
    latency: float = Field(...)
    model: str = Field(...)
    finish_reason: Optional[str] = Field(None)
    error: Optional[str] = Field(None)


class EvaluationResult(BaseModel):
    problem_id: Any = Field(...)
    question: str = Field(...)
    formatted_prompt: Optional[str] = Field(None)
    ground_truth: Any = Field(...)

    model_response: str = Field(...)
    extracted_answer: Optional[Any] = Field(None)

    correct: bool = Field(...)

    tokens: TokenUsage = Field(...)
    latency: float = Field(...)

    error: Optional[str] = Field(None)
    extraction_method: Optional[str] = Field(None)
    finish_reason: Optional[str] = Field(None)

    tests_passed: Optional[int] = Field(None)
    tests_total: Optional[int] = Field(None)
    execution_error: Optional[str] = Field(None)


class ExperimentSummary(BaseModel):
    total_problems: int = Field(...)
    correct_count: int = Field(...)
    accuracy: float = Field(...)

    total_tokens: int = Field(...)
    total_prompt_tokens: int = Field(...)
    total_answer_tokens: int = Field(...)
    total_reasoning_tokens: int = Field(...)
    total_output_tokens: int = Field(...)

    avg_tokens_per_problem: float = Field(...)
    avg_latency: float = Field(...)

    total_duration: float = Field(...)

    error_count: int = Field(0)
    ock_score: Optional[float] = Field(None)

    def __init__(self, **data):
        if 'total_completion_tokens' in data and 'total_answer_tokens' not in data:
            data['total_answer_tokens'] = data.pop('total_completion_tokens')
        if 'total_output_tokens' not in data or data.get('total_output_tokens') == 0:
            data['total_output_tokens'] = data.get('total_answer_tokens', 0) + data.get('total_reasoning_tokens', 0)
        super().__init__(**data)


class ExperimentResult(BaseModel):
    config: BenchmarkConfig = Field(...)
    results: List[EvaluationResult] = Field(...)
    summary: ExperimentSummary = Field(...)

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    dataset_name: str = Field(...)

    def save_to_file(self, filepath: str):
        import json

        from ..utils.request_overrides import redact_config

        data = self.model_dump()
        data['config'] = redact_config(data.get('config', {}))
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ExperimentResult":
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Result files written by older versions persist removed config fields.
        # `enable_thinking` is fully removed, so always drop it. `reasoning_effort`
        # is still a valid field for non-chat providers, so only drop it for a
        # chat_completion config (where it is removed and would fail validation);
        # otherwise keep it so non-chat results round-trip their configuration.
        config = data.get('config')
        if isinstance(config, dict):
            config.pop('enable_thinking', None)
            if config.get('provider') == 'chat_completion':
                config.pop('reasoning_effort', None)
        return cls(**data)
