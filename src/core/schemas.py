"""Pydantic schemas for OckBench."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Result/cache schema version. Bump when the on-disk result or cache shape
# changes in a way that affects reproducibility or aggregation.
SCHEMA_VERSION = "2.0"


class RequestOverrides(BaseModel):
    """User-driven overrides applied to the outgoing request.

    ``set`` maps a dotted path to a JSON-typed value (deep create/replace);
    ``unset`` lists dotted paths to remove (deep delete). Applied in that order.
    """

    set: Dict[str, Any] = Field(default_factory=dict)
    unset: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_paths(self) -> 'RequestOverrides':
        for path in list(self.set.keys()) + self.unset:
            if not path.strip():
                raise ValueError("request override paths must be non-empty strings")
        return self


class JudgeConfig(BaseModel):
    """Configuration for the math LLM judge (an OpenAI-compatible endpoint).

    The judge scores the extracted answer block. ``request_overrides`` lets the
    judge call be shaped the same way model requests are (e.g. to disable a local
    thinking model's reasoning). Credentials are masked in saved provenance.
    """

    model: str = Field(...)
    base_url: Optional[str] = Field(None)
    api_key: Optional[str] = Field(None)
    timeout: int = Field(60, gt=0)
    max_tokens: int = Field(500, gt=0)
    request_overrides: RequestOverrides = Field(default_factory=RequestOverrides)


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark experiment."""

    dataset_path: str = Field(...)
    dataset_name: Optional[str] = Field(None)
    dataset_split: Optional[str] = Field(None)

    # Free-form provider name resolved through the provider registry, so an
    # externally registered provider is selectable without editing this schema.
    provider: str = Field(...)
    model: str = Field(...)
    base_url: Optional[str] = Field(None)
    api_key: Optional[str] = Field(None)

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(None, gt=0)
    max_context_window: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)

    # All reasoning/thinking placement is expressed uniformly through
    # request_overrides for every provider (no per-provider hard-coding).
    request_overrides: RequestOverrides = Field(default_factory=RequestOverrides)

    # Math scoring uses this judge; it is required at run start when the
    # evaluator is the math judge (enforced where the evaluator is built).
    judge: Optional[JudgeConfig] = Field(None)

    concurrency: int = Field(5, gt=0)
    timeout: int = Field(120, gt=0)
    wall_clock_timeout: Optional[int] = Field(None, gt=0)
    max_retries: int = Field(3, gt=0)

    evaluator_type: str = Field("math")
    execution_timeout: int = Field(5, gt=0)
    include_challenge_tests: bool = Field(True)

    experiment_name: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)

    @model_validator(mode='before')
    @classmethod
    def _reject_legacy_reasoning_fields(cls, data: Any) -> Any:
        # Reasoning/thinking is now configured uniformly through request_overrides
        # for every provider. Reject the removed `reasoning_effort` /
        # `enable_thinking` keys loudly so an old config does not silently run with
        # a different request shape (pydantic would otherwise ignore them). A null
        # leftover from an older serialized config is not an intentional setting
        # and is ignored.
        if not isinstance(data, dict):
            return data
        legacy = {
            'reasoning_effort': (
                "set reasoning effort via request_overrides — e.g. chat_completion "
                "'reasoning_effort', openai-responses 'reasoning.effort', anthropic "
                "'output_config.effort', gemini 'config.thinking_config.thinking_budget'"
            ),
            'enable_thinking': (
                "set thinking via request_overrides — e.g. "
                "'extra_body.chat_template_kwargs.enable_thinking'"
            ),
        }
        present = [key for key in legacy if data.get(key) is not None]
        if present:
            hints = "; ".join(f"{key} -> {legacy[key]}" for key in present)
            raise ValueError(
                f"removed config field(s) {present}: reasoning/thinking is now configured "
                f"through request_overrides for all providers. Migrate: {hints}"
            )
        return data

    @model_validator(mode='after')
    def validate_config(self) -> 'BenchmarkConfig':
        if self.provider == 'chat_completion':
            if not self.api_key:
                raise ValueError("chat_completion provider requires --api-key")
            if not self.base_url:
                raise ValueError("chat_completion provider requires --base-url")

        has_max_output = self.max_output_tokens is not None
        has_max_context = self.max_context_window is not None

        if has_max_output and has_max_context:
            raise ValueError("max_output_tokens and max_context_window are mutually exclusive.")
        if not has_max_output and not has_max_context:
            raise ValueError("Either max_output_tokens or max_context_window must be set.")
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
    # True when the answer/reasoning split rests on a tokenizer estimate rather
    # than an exact provider count (Anthropic count_tokens fallback, or a
    # repaired impossible relay split). Diagnostic only — never enters scoring.
    # Additive: rows written before this field existed read back as False.
    answer_tokens_estimated: bool = Field(False)
    # Tokens the provider billed only in its total (attributed to neither
    # prompt nor completion — hidden thinking on some relays). They are folded
    # into output/reasoning at normalization; this count preserves the
    # provider's original attribution. Diagnostic only — never enters scoring.
    # Additive: rows written before this field existed read back as 0.
    unattributed_tokens: int = Field(0)

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

    # Generation-side failure (provider/transport). Never carries evaluator
    # failures: a judge outage must not overwrite the generation terminal
    # state (finish_reason/tokens) that recovery decisions key on.
    error: Optional[str] = Field(None)
    # Evaluator/judge-side failure with a successful generation. Additive:
    # rows written before this field existed carry judge failures in `error`
    # and read back as None here; resume handles both shapes.
    evaluator_error: Optional[str] = Field(None)
    extraction_method: Optional[str] = Field(None)
    finish_reason: Optional[str] = Field(None)

    # Full outcome for math: the judge's free-text rationale (cached so a resume
    # never re-invokes the judge).
    judge_reasoning: Optional[str] = Field(None)

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
    schema_version: str = Field(default=SCHEMA_VERSION)
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
        # Historical result files may carry the removed reasoning fields as
        # provenance; drop them so old results still load. (Running configs that
        # still set them are rejected at construction — see the validator above.)
        config = data.get('config')
        if isinstance(config, dict):
            config.pop('reasoning_effort', None)
            config.pop('enable_thinking', None)
        return cls(**data)
