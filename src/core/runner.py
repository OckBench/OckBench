"""Main benchmark runner with concurrent API calls and result aggregation."""
import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import List, Optional, Set

from tqdm.asyncio import tqdm as atqdm

from ..evaluators.math_eval import get_evaluator
from ..loaders.base import get_loader
from ..models.anthropic_api import AnthropicClient
from ..models.base import BaseModelClient
from ..models.gemini_api import GeminiClient
from ..models.openai_api import OpenAIClient
from ..models.openai_responses_api import OpenAIResponsesClient
from ..utils.logger import get_experiment_filename, setup_logger
from .config import load_config
from .schemas import BenchmarkConfig, EvaluationResult, ExperimentResult, ExperimentSummary, Problem, TokenUsage

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates loading, calling, evaluating, caching, and aggregating."""

    def __init__(self, config: BenchmarkConfig, cache_path: Optional[str] = None):
        self.config = config
        self.cache_path = cache_path
        self.client: Optional[BaseModelClient] = None
        self.evaluator = None
        self.problems: List[Problem] = []
        self._cached_results: List[EvaluationResult] = []

    def _load_cache(self) -> Set:
        """Load completed problem IDs from cache file."""
        cache_file = Path(self.cache_path)
        if not cache_file.exists():
            return set()

        completed_ids = set()
        self._cached_results = []
        line_num = 0
        for line in cache_file.read_text(encoding='utf-8').strip().split('\n'):
            line_num += 1
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                result = EvaluationResult(**data)
                if not result.error:
                    completed_ids.add(result.problem_id)
                    self._cached_results.append(result)
                else:
                    logger.debug(f"Cache line {line_num}: problem {result.problem_id} had error, will retry")
            except Exception as e:
                logger.warning(f"Cache line {line_num}: failed to parse, skipping: {e}")

        logger.info(f"Loaded {len(completed_ids)} completed results from cache")
        return completed_ids

    def _append_to_cache(self, result: EvaluationResult):
        if not self.cache_path:
            return
        cache_file = Path(self.cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(result.model_dump_json() + '\n')
            f.flush()

    def _create_client(self) -> BaseModelClient:
        client_kwargs = {
            'model': self.config.model,
            'api_key': self.config.api_key,
            'base_url': self.config.base_url,
            'timeout': self.config.timeout,
            'max_retries': self.config.max_retries,
        }

        if self.config.provider == 'chat_completion':
            return OpenAIClient(**client_kwargs)
        elif self.config.provider == 'openai-responses':
            return OpenAIResponsesClient(**client_kwargs)
        elif self.config.provider == 'anthropic':
            return AnthropicClient(**client_kwargs)
        elif self.config.provider == 'gemini':
            return GeminiClient(**client_kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _calculate_max_output_tokens(self, prompt: str) -> int:
        if self.config.max_context_window is not None:
            from ..utils.token_counter import estimate_tokens

            try:
                input_tokens = estimate_tokens(prompt, self.config.model)
            except Exception as e:
                logger.warning(f"Failed to estimate tokens, using rough estimate: {e}")
                input_tokens = len(prompt) // 4

            safety_buffer = 256
            max_output = self.config.max_context_window - input_tokens - safety_buffer
            max_output = max(max_output, 100)

            logger.debug(
                f"Dynamic max_output_tokens: {max_output} "
                f"(context: {self.config.max_context_window}, input: {input_tokens})"
            )
            return max_output
        else:
            return self.config.max_output_tokens

    async def _process_single_problem(
        self,
        problem: Problem,
        semaphore: asyncio.Semaphore,
        pbar: Optional[atqdm] = None
    ) -> EvaluationResult:
        from ..utils.prompt_formatter import format_prompt

        test_cases_for_prompt = None
        if self.config.evaluator_type == "code":
            test_cases_for_prompt = problem.metadata.get('test_list', [])

        formatted_prompt = format_prompt(
            problem=problem.problem,
            evaluator_type=self.config.evaluator_type,
            test_cases=test_cases_for_prompt
        )

        async with semaphore:
            try:
                max_output_tokens = self._calculate_max_output_tokens(formatted_prompt)

                response = await self.client.generate(
                    prompt=formatted_prompt,
                    temperature=self.config.temperature,
                    max_output_tokens=max_output_tokens,
                    reasoning_effort=self.config.reasoning_effort,
                    top_p=self.config.top_p,
                    enable_thinking=self.config.enable_thinking
                )

                if response.error:
                    logger.error(f"Error for problem {problem.id}: {response.error}")
                    result = EvaluationResult(
                        problem_id=problem.id,
                        question=problem.problem,
                        formatted_prompt=formatted_prompt,
                        ground_truth=problem.answer,
                        model_response=response.text or "",
                        extracted_answer=None,
                        correct=False,
                        tokens=response.tokens,
                        latency=response.latency,
                        error=response.error,
                        extraction_method="error",
                        finish_reason=response.finish_reason,
                    )
                else:
                    if self.config.evaluator_type == "code":
                        eval_result = self.evaluator.evaluate(response.text, problem.metadata.get('test_cases', []))
                    else:
                        eval_result = self.evaluator.evaluate(response.text, problem.answer)

                    result = EvaluationResult(
                        problem_id=problem.id,
                        question=problem.problem,
                        formatted_prompt=formatted_prompt,
                        ground_truth=problem.answer,
                        model_response=response.text,
                        extracted_answer=eval_result.extracted_answer,
                        correct=eval_result.is_correct,
                        tokens=response.tokens,
                        latency=response.latency,
                        extraction_method=eval_result.extraction_method,
                        tests_passed=eval_result.tests_passed,
                        tests_total=eval_result.tests_total,
                        execution_error=eval_result.execution_error,
                        finish_reason=response.finish_reason,
                    )

                self._append_to_cache(result)
                if pbar:
                    pbar.update(1)
                return result

            except Exception as e:
                logger.error(f"Exception processing problem {problem.id}: {e}")
                result = EvaluationResult(
                    problem_id=problem.id,
                    question=problem.problem,
                    formatted_prompt=formatted_prompt,
                    ground_truth=problem.answer,
                    model_response="",
                    extracted_answer=None,
                    correct=False,
                    tokens=TokenUsage(
                        prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                        output_tokens=0, total_tokens=0,
                    ),
                    latency=0,
                    error=str(e),
                    extraction_method="exception"
                )
                self._append_to_cache(result)
                if pbar:
                    pbar.update(1)
                return result

    async def _run_benchmark_async(self) -> List[EvaluationResult]:
        semaphore = asyncio.Semaphore(self.config.concurrency)

        pbar = atqdm(
            total=len(self.problems),
            desc=f"Running {self.config.model} on {len(self.problems)} problems",
            unit="problem"
        )

        tasks = [
            self._process_single_problem(problem, semaphore, pbar)
            for problem in self.problems
        ]
        results = await asyncio.gather(*tasks)
        pbar.close()

        return results

    def _compute_summary(self, results: List[EvaluationResult], duration: float) -> ExperimentSummary:
        total_problems = len(results)
        correct_count = sum(1 for r in results if r.correct)
        accuracy = (correct_count / total_problems * 100) if total_problems > 0 else 0

        total_prompt_tokens = sum(r.tokens.prompt_tokens for r in results)
        total_answer_tokens = sum(r.tokens.answer_tokens for r in results)
        total_reasoning_tokens = sum(r.tokens.reasoning_tokens for r in results)
        total_output_tokens = sum(r.tokens.output_tokens for r in results)
        total_tokens = sum(r.tokens.total_tokens for r in results)

        avg_tokens = total_tokens / total_problems if total_problems > 0 else 0
        avg_latency = sum(r.latency for r in results) / total_problems if total_problems > 0 else 0

        error_count = sum(1 for r in results if r.error)
        ock_score = accuracy - 10 * math.log(avg_tokens / 10000 + 1)

        return ExperimentSummary(
            total_problems=total_problems,
            correct_count=correct_count,
            accuracy=accuracy,
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt_tokens,
            total_answer_tokens=total_answer_tokens,
            total_reasoning_tokens=total_reasoning_tokens,
            total_output_tokens=total_output_tokens,
            avg_tokens_per_problem=avg_tokens,
            avg_latency=avg_latency,
            total_duration=duration,
            error_count=error_count,
            ock_score=ock_score,
        )

    def run(self) -> ExperimentResult:
        """Run complete benchmark experiment."""
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

        logger.info("Loading dataset...")
        loader = get_loader(filepath=self.config.dataset_path)
        self.problems = loader.load()
        logger.info(f"Loaded {len(self.problems)} problems")

        if self.cache_path:
            completed_ids = self._load_cache()
            if completed_ids:
                self.problems = [p for p in self.problems if p.id not in completed_ids]
                logger.info(f"Resuming: {len(completed_ids)} cached, {len(self.problems)} remaining")

        logger.info("Initializing API client...")
        self.client = self._create_client()

        logger.info("Initializing evaluator...")
        evaluator_kwargs = {}
        if self.config.evaluator_type == "code":
            evaluator_kwargs['timeout'] = self.config.execution_timeout
        self.evaluator = get_evaluator(self.config.evaluator_type, **evaluator_kwargs)

        if self.problems:
            logger.info("Running benchmark...")
            new_results = asyncio.run(self._run_benchmark_async())
        else:
            logger.info("All problems already cached, no work to do")
            new_results = []

        results = self._cached_results + list(new_results)
        duration = time.time() - start_time
        summary = self._compute_summary(results, duration)

        logger.info("=" * 80)
        logger.info("Experiment Complete!")
        logger.info(f"Accuracy: {summary.accuracy:.2f}% ({summary.correct_count}/{summary.total_problems})")
        logger.info(f"Total Tokens: {summary.total_tokens:,}")
        logger.info(f"  Prompt: {summary.total_prompt_tokens:,}")
        logger.info(f"  Answer: {summary.total_answer_tokens:,}")
        logger.info(f"  Reasoning: {summary.total_reasoning_tokens:,}")
        logger.info(f"  Output: {summary.total_output_tokens:,}")
        logger.info(f"Avg Tokens/Problem: {summary.avg_tokens_per_problem:.1f}")
        logger.info(f"OckScore: {summary.ock_score:.2f}")
        logger.info(f"Duration: {summary.total_duration:.2f}s")
        if summary.error_count > 0:
            logger.warning(f"Errors: {summary.error_count}")
        logger.info("=" * 80)

        dataset_name = self.config.dataset_name or Path(self.config.dataset_path).stem

        return ExperimentResult(
            config=self.config,
            results=results,
            summary=summary,
            dataset_name=dataset_name,
        )


def run_benchmark(
    config_path: Optional[str] = None,
    output_dir: str = "results",
    log_dir: Optional[str] = None,
    cache: Optional[str] = None,
    **config_overrides
) -> ExperimentResult:
    """Main entry point for running benchmarks."""
    config_overrides.pop('cache', None)
    config = load_config(config_path, **config_overrides)

    dataset_name = config.dataset_name or Path(config.dataset_path).stem
    from ..utils.logger import get_log_filename

    if log_dir:
        log_file = Path(log_dir) / get_log_filename(dataset_name, config.model)
    else:
        log_file = None

    setup_logger(log_file=str(log_file) if log_file else None)

    runner = BenchmarkRunner(config, cache_path=cache)
    experiment = runner.run()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / get_experiment_filename(
        experiment.dataset_name,
        config.model
    )

    experiment.save_to_file(str(result_file))
    logger.info(f"Results saved to: {result_file}")

    return experiment
