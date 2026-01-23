"""
Pydantic schemas for OckBench benchmarking tool.
"""
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark experiment."""
    
    # Dataset config
    dataset_path: str = Field(..., description="Path to the dataset file")
    dataset_name: Optional[str] = Field(None, description="Name of the dataset for logging")
    
    # Model config
    provider: Literal["openai", "gemini", "generic"] = Field(..., description="API provider type")
    model: str = Field(..., description="Model name/identifier")
    base_url: Optional[str] = Field(None, description="Base URL for API (for generic/local providers)")
    api_key: Optional[str] = Field(None, description="API key (can be from env)")
    
    # Generation parameters
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_output_tokens: Optional[int] = Field(4096, gt=0, description="Maximum output tokens (can be omitted if max_context_window is set)")
    max_context_window: Optional[int] = Field(None, gt=0, description="Maximum context window (input + output). If set, max_output_tokens will be calculated dynamically")
    reasoning_effort: Optional[str] = Field(None, description="Reasoning effort level (for o1/o3 models)")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    enable_thinking: Optional[bool] = Field(None, description="Enable/disable thinking mode for Qwen3 models (None=use model default)")
    
    # Runtime config
    concurrency: int = Field(5, gt=0, description="Number of concurrent API requests")
    timeout: int = Field(120, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(3, gt=0, description="Maximum number of retries per request")
    
    # Evaluation config
    evaluator_type: str = Field("math", description="Type of evaluator to use")
    enforce_output_format: bool = Field(False, description="Add instructions to prompt to enforce consistent output format")
    custom_format_instruction: Optional[str] = Field(None, description="Custom format instruction (overrides default)")
    
    # Code evaluation specific
    execution_timeout: int = Field(5, gt=0, description="Timeout for code execution in seconds")
    include_challenge_tests: bool = Field(True, description="Include challenge tests in code evaluation")
    
    # Optional metadata
    experiment_name: Optional[str] = Field(None, description="Custom experiment name")
    notes: Optional[str] = Field(None, description="Additional notes about the experiment")


class Problem(BaseModel):
    """Represents a single problem from a dataset."""
    
    problem: str = Field(..., description="Problem text/question")
    answer: Any = Field(..., description="Ground truth answer")
    id: Any = Field(..., description="Problem identifier")
    
    # Optional fields for additional context
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional problem metadata")


class TokenUsage(BaseModel):
    """Token usage information from API response."""
    
    prompt_tokens: int = Field(0, description="Input/prompt tokens")
    answer_tokens: int = Field(0, description="Answer/output tokens")
    reasoning_tokens: int = Field(0, description="Reasoning tokens (for o1/o3 models)")
    output_tokens: int = Field(0, description="Total output tokens (reasoning + answer)")
    total_tokens: int = Field(0, description="Total tokens used")
    
    def __init__(self, **data):
        # Handle backward compatibility: old files might have completion_tokens
        if 'completion_tokens' in data and 'answer_tokens' not in data:
            data['answer_tokens'] = data.pop('completion_tokens')
        super().__init__(**data)
        # Auto-calculate output_tokens if not provided
        if self.output_tokens == 0:
            self.output_tokens = self.reasoning_tokens + self.answer_tokens
        # Auto-calculate total if not provided
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.answer_tokens + self.reasoning_tokens


class ModelResponse(BaseModel):
    """Response from model API."""
    
    text: str = Field(..., description="Generated text response")
    tokens: TokenUsage = Field(..., description="Token usage information")
    latency: float = Field(..., description="Response latency in seconds")
    model: str = Field(..., description="Model that generated the response")
    
    # Optional fields
    finish_reason: Optional[str] = Field(None, description="Reason for completion")
    error: Optional[str] = Field(None, description="Error message if request failed")


class EvaluationResult(BaseModel):
    """Result of evaluating a single problem."""
    
    problem_id: Any = Field(..., description="Problem identifier")
    question: str = Field(..., description="Original question")
    formatted_prompt: Optional[str] = Field(None, description="Formatted prompt sent to model (includes format instructions, test cases, etc.)")
    ground_truth: Any = Field(..., description="Ground truth answer")
    
    model_response: str = Field(..., description="Full model response text")
    extracted_answer: Optional[Any] = Field(None, description="Extracted answer from response")
    
    correct: bool = Field(..., description="Whether the answer is correct")
    
    tokens: TokenUsage = Field(..., description="Token usage for this problem")
    latency: float = Field(..., description="Time taken for this problem")
    
    error: Optional[str] = Field(None, description="Error message if evaluation failed")
    extraction_method: Optional[str] = Field(None, description="Method used to extract answer")
    
    # Code evaluation specific fields
    tests_passed: Optional[int] = Field(None, description="Number of tests passed (code problems)")
    tests_total: Optional[int] = Field(None, description="Total number of tests (code problems)")
    execution_error: Optional[str] = Field(None, description="Code execution error details")


class ExperimentSummary(BaseModel):
    """Summary statistics for an experiment."""
    
    total_problems: int = Field(..., description="Total number of problems")
    correct_count: int = Field(..., description="Number of correct answers")
    accuracy: float = Field(..., description="Accuracy percentage")
    
    total_tokens: int = Field(..., description="Total tokens used")
    total_prompt_tokens: int = Field(..., description="Total prompt tokens")
    total_answer_tokens: int = Field(..., description="Total answer tokens")
    total_reasoning_tokens: int = Field(..., description="Total reasoning tokens")
    total_output_tokens: int = Field(..., description="Total output tokens (reasoning + answer)")
    
    avg_tokens_per_problem: float = Field(..., description="Average tokens per problem")
    avg_latency: float = Field(..., description="Average latency per problem")
    
    total_duration: float = Field(..., description="Total experiment duration in seconds")
    
    error_count: int = Field(0, description="Number of errors encountered")
    
    def __init__(self, **data):
        # Handle backward compatibility: old files might have total_completion_tokens
        if 'total_completion_tokens' in data and 'total_answer_tokens' not in data:
            data['total_answer_tokens'] = data.pop('total_completion_tokens')
        # Calculate total_output_tokens if not provided (for backward compatibility)
        if 'total_output_tokens' not in data or data.get('total_output_tokens') == 0:
            total_answer = data.get('total_answer_tokens', 0)
            total_reasoning = data.get('total_reasoning_tokens', 0)
            data['total_output_tokens'] = total_answer + total_reasoning
        super().__init__(**data)


class ExperimentResult(BaseModel):
    """Complete result of a benchmark experiment."""
    
    config: BenchmarkConfig = Field(..., description="Experiment configuration")
    results: List[EvaluationResult] = Field(..., description="Per-problem results")
    summary: ExperimentSummary = Field(..., description="Summary statistics")
    
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Experiment timestamp")
    dataset_name: str = Field(..., description="Name of the dataset")
    
    def save_to_file(self, filepath: str):
        """Save experiment result to JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "ExperimentResult":
        """Load experiment result from JSON file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

