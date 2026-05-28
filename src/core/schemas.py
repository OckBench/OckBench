"""Pydantic schemas for OckBench."""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


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
    reasoning_effort: Optional[str] = Field(None)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    enable_thinking: Optional[bool] = Field(None)

    concurrency: int = Field(5, gt=0)
    timeout: int = Field(120, gt=0)
    max_retries: int = Field(3, gt=0)

    evaluator_type: str = Field("math")
    execution_timeout: int = Field(5, gt=0)
    include_challenge_tests: bool = Field(True)

    experiment_name: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)

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
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ExperimentResult":
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
