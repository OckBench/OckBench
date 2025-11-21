"""
Main benchmark runner with concurrent API calls and result aggregation.
"""
import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional
from tqdm.asyncio import tqdm as atqdm

from .config import load_config
from .schemas import (
    BenchmarkConfig, Problem, EvaluationResult, 
    ExperimentResult, ExperimentSummary
)
from ..loaders.base import get_loader
from ..models.base import BaseModelClient
from ..models.openai_api import OpenAIClient
from ..models.gemini_api import GeminiClient
from ..evaluators.math_eval import get_evaluator
from ..utils.logger import setup_logger, get_experiment_filename


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main runner for benchmark experiments.
    
    Orchestrates:
    - Loading data and config
    - Creating model client
    - Running concurrent API calls
    - Evaluating responses
    - Aggregating and saving results
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.client: Optional[BaseModelClient] = None
        self.evaluator = None
        self.problems: List[Problem] = []
    
    def _create_client(self) -> BaseModelClient:
        """
        Create appropriate API client based on provider.
        
        Returns:
            BaseModelClient: Initialized API client
        """
        client_kwargs = {
            'model': self.config.model,
            'api_key': self.config.api_key,
            'base_url': self.config.base_url,
            'timeout': self.config.timeout,
            'max_retries': self.config.max_retries,
        }
        
        if self.config.provider == 'openai':
            return OpenAIClient(**client_kwargs)
        elif self.config.provider == 'gemini':
            return GeminiClient(**client_kwargs)
        elif self.config.provider == 'generic':
            # Generic uses OpenAI-compatible API
            if not self.config.base_url:
                raise ValueError("base_url is required for generic provider")
            return OpenAIClient(**client_kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _calculate_max_output_tokens(self, prompt: str) -> int:
        """
        Calculate max_output_tokens based on config and prompt length.
        
        If max_context_window is set, calculates dynamically.
        Otherwise, uses the configured max_output_tokens.
        
        Args:
            prompt: Input prompt text
        
        Returns:
            int: Maximum output tokens for this request
        """
        if self.config.max_context_window is not None:
            # Estimate input tokens
            from ..utils.token_counter import estimate_tokens
            
            try:
                input_tokens = estimate_tokens(prompt, self.config.model)
            except Exception as e:
                logger.warning(f"Failed to estimate tokens, using rough estimate: {e}")
                # Rough estimate: ~4 chars per token
                input_tokens = len(prompt) // 4
            
            # Calculate available output tokens with safety buffer (100 tokens)
            safety_buffer = 100
            max_output = self.config.max_context_window - input_tokens - safety_buffer
            
            # Ensure we have at least some minimum output space
            min_output = 100
            max_output = max(max_output, min_output)
            
            logger.debug(
                f"Dynamic max_output_tokens: {max_output} "
                f"(context: {self.config.max_context_window}, input: {input_tokens})"
            )
            
            return max_output
        else:
            # Use configured value
            return self.config.max_output_tokens or 4096
    
    async def _process_single_problem(
        self,
        problem: Problem,
        semaphore: asyncio.Semaphore,
        pbar: Optional[atqdm] = None
    ) -> EvaluationResult:
        """
        Process a single problem with rate limiting.
        
        Args:
            problem: Problem to process
            semaphore: Semaphore for concurrency control
            pbar: Progress bar (optional)
        
        Returns:
            EvaluationResult: Evaluation result for this problem
        """
        async with semaphore:
            try:
                # Calculate max_output_tokens (dynamic if max_context_window is set)
                max_output_tokens = self._calculate_max_output_tokens(problem.problem)
                
                # Generate response
                response = await self.client.generate(
                    prompt=problem.problem,
                    temperature=self.config.temperature,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=self.config.reasoning_effort,
                    top_p=self.config.top_p
                )
                
                # Check for API error
                if response.error:
                    logger.error(f"Error for problem {problem.id}: {response.error}")
                    result = EvaluationResult(
                        problem_id=problem.id,
                        question=problem.problem,
                        ground_truth=problem.answer,
                        model_response=response.text or "",
                        extracted_answer=None,
                        correct=False,
                        tokens=response.tokens,
                        latency=response.latency,
                        error=response.error,
                        extraction_method="error"
                    )
                else:
                    # Evaluate response
                    is_correct, extracted_answer, method = self.evaluator.evaluate(
                        response.text,
                        problem.answer
                    )
                    
                    result = EvaluationResult(
                        problem_id=problem.id,
                        question=problem.problem,
                        ground_truth=problem.answer,
                        model_response=response.text,
                        extracted_answer=extracted_answer,
                        correct=is_correct,
                        tokens=response.tokens,
                        latency=response.latency,
                        extraction_method=method
                    )
                
                if pbar:
                    pbar.update(1)
                
                return result
                
            except Exception as e:
                logger.error(f"Exception processing problem {problem.id}: {e}")
                
                # Return error result
                from ..core.schemas import TokenUsage
                result = EvaluationResult(
                    problem_id=problem.id,
                    question=problem.problem,
                    ground_truth=problem.answer,
                    model_response="",
                    extracted_answer=None,
                    correct=False,
                    tokens=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        reasoning_tokens=0,
                        total_tokens=0
                    ),
                    latency=0,
                    error=str(e),
                    extraction_method="exception"
                )
                
                if pbar:
                    pbar.update(1)
                
                return result
    
    async def _run_benchmark_async(self) -> List[EvaluationResult]:
        """
        Run benchmark on all problems concurrently.
        
        Returns:
            List[EvaluationResult]: Results for all problems
        """
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.concurrency)
        
        # Create progress bar
        pbar = atqdm(
            total=len(self.problems),
            desc=f"Running {self.config.model} on {len(self.problems)} problems",
            unit="problem"
        )
        
        # Process all problems concurrently
        tasks = [
            self._process_single_problem(problem, semaphore, pbar)
            for problem in self.problems
        ]
        
        results = await asyncio.gather(*tasks)
        
        pbar.close()
        
        return results
    
    def _compute_summary(
        self,
        results: List[EvaluationResult],
        duration: float
    ) -> ExperimentSummary:
        """
        Compute summary statistics from results.
        
        Args:
            results: List of evaluation results
            duration: Total experiment duration
        
        Returns:
            ExperimentSummary: Summary statistics
        """
        total_problems = len(results)
        correct_count = sum(1 for r in results if r.correct)
        accuracy = (correct_count / total_problems * 100) if total_problems > 0 else 0
        
        total_prompt_tokens = sum(r.tokens.prompt_tokens for r in results)
        total_completion_tokens = sum(r.tokens.completion_tokens for r in results)
        total_reasoning_tokens = sum(r.tokens.reasoning_tokens for r in results)
        total_tokens = sum(r.tokens.total_tokens for r in results)
        
        avg_tokens = total_tokens / total_problems if total_problems > 0 else 0
        avg_latency = sum(r.latency for r in results) / total_problems if total_problems > 0 else 0
        
        error_count = sum(1 for r in results if r.error)
        
        return ExperimentSummary(
            total_problems=total_problems,
            correct_count=correct_count,
            accuracy=accuracy,
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_reasoning_tokens=total_reasoning_tokens,
            avg_tokens_per_problem=avg_tokens,
            avg_latency=avg_latency,
            total_duration=duration,
            error_count=error_count
        )
    
    def run(self) -> ExperimentResult:
        """
        Run complete benchmark experiment.
        
        Returns:
            ExperimentResult: Complete experiment results
        """
        logger.info("=" * 80)
        logger.info("Starting OckBench Experiment")
        logger.info("=" * 80)
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"Provider: {self.config.provider}")
        logger.info(f"Temperature: {self.config.temperature}")
        logger.info(f"Max Output Tokens: {self.config.max_output_tokens}")
        logger.info(f"Concurrency: {self.config.concurrency}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Load dataset
        logger.info("Loading dataset...")
        loader = get_loader(filepath=self.config.dataset_path)
        self.problems = loader.load()
        logger.info(f"Loaded {len(self.problems)} problems")
        
        # Create client
        logger.info("Initializing API client...")
        self.client = self._create_client()
        
        # Create evaluator
        logger.info("Initializing evaluator...")
        self.evaluator = get_evaluator(self.config.evaluator_type)
        
        # Run benchmark
        logger.info("Running benchmark...")
        results = asyncio.run(self._run_benchmark_async())
        
        # Compute summary
        duration = time.time() - start_time
        summary = self._compute_summary(results, duration)
        
        # Log summary
        logger.info("=" * 80)
        logger.info("Experiment Complete!")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {summary.accuracy:.2f}% ({summary.correct_count}/{summary.total_problems})")
        logger.info(f"Total Tokens: {summary.total_tokens:,}")
        logger.info(f"  - Prompt: {summary.total_prompt_tokens:,}")
        logger.info(f"  - Completion: {summary.total_completion_tokens:,}")
        logger.info(f"  - Reasoning: {summary.total_reasoning_tokens:,}")
        logger.info(f"Avg Tokens/Problem: {summary.avg_tokens_per_problem:.1f}")
        logger.info(f"Avg Latency: {summary.avg_latency:.2f}s")
        logger.info(f"Total Duration: {summary.total_duration:.2f}s")
        if summary.error_count > 0:
            logger.warning(f"Errors: {summary.error_count}")
        logger.info("=" * 80)
        
        # Get dataset name
        dataset_name = self.config.dataset_name or Path(self.config.dataset_path).stem
        
        # Create experiment result
        experiment = ExperimentResult(
            config=self.config,
            results=results,
            summary=summary,
            dataset_name=dataset_name
        )
        
        return experiment


def run_benchmark(
    config_path: Optional[str] = None,
    output_dir: str = "results",
    log_dir: Optional[str] = None,
    **config_overrides
) -> ExperimentResult:
    """
    Main entry point for running benchmarks.
    
    Args:
        config_path: Path to config YAML file (optional)
        output_dir: Directory to save results
        log_dir: Directory to save logs (optional)
        **config_overrides: Config parameters to override
    
    Returns:
        ExperimentResult: Complete experiment results
    """
    # Load config
    config = load_config(config_path, **config_overrides)
    
    # Setup logging
    dataset_name = config.dataset_name or Path(config.dataset_path).stem
    from ..utils.logger import get_log_filename
    
    if log_dir:
        log_file = Path(log_dir) / get_log_filename(dataset_name, config.model)
    else:
        log_file = None
    
    setup_logger(log_file=str(log_file) if log_file else None)
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    experiment = runner.run()
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    result_file = output_path / get_experiment_filename(
        experiment.dataset_name,
        config.model
    )
    
    experiment.save_to_file(str(result_file))
    logger.info(f"Results saved to: {result_file}")
    
    return experiment

