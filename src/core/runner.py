"""Main benchmark runner with concurrent API calls and result aggregation."""
import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm as atqdm

from ..evaluators import get_evaluator
from ..evaluators.base import EvalResult
from ..loaders.base import get_loader
from ..models import create_provider
from ..models.base import BaseModelClient
from ..utils.logger import get_experiment_filename, setup_logger
from .cache import RunCache
from .config import load_config
from .schemas import BenchmarkConfig, EvaluationResult, ExperimentResult, ExperimentSummary, Problem, TokenUsage
from .scoring import summarize

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates loading, calling, evaluating, caching, and aggregating."""

    def __init__(self, config: BenchmarkConfig, cache_path: Optional[str] = None):
        self.config = config
        self.cache_path = cache_path
        self.cache: Optional[RunCache] = None
        self.client: Optional[BaseModelClient] = None
        self.evaluator = None
        self.problems: List[Problem] = []
        self.rejudge_items: List[tuple[Problem, EvaluationResult]] = []
        self._client_closed = False

    def _create_client(self) -> BaseModelClient:
        """Resolve and construct the provider through the registry (no if/elif)."""
        return create_provider(
            self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            wall_clock_timeout=self.config.wall_clock_timeout,
            max_retries=self.config.max_retries,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            request_overrides=self.config.request_overrides,
        )

    async def _run_and_close_client(self) -> List[EvaluationResult]:
        try:
            return await self._run_benchmark_async()
        finally:
            await self.client.aclose()
            self._client_closed = True

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
        return self.config.max_output_tokens

    @staticmethod
    def _result_from_eval(
        problem_fields: Dict[str, Any],
        eval_result: EvalResult,
        *,
        model_response: str,
        tokens: TokenUsage,
        latency: float,
        finish_reason: Optional[str],
    ) -> EvaluationResult:
        """The single eval→result field mapping, shared by fresh runs and rejudges."""
        return EvaluationResult(
            **problem_fields,
            model_response=model_response,
            extracted_answer=eval_result.extracted_answer,
            correct=eval_result.is_correct,
            tokens=tokens,
            latency=latency,
            extraction_method=eval_result.extraction_method,
            judge_reasoning=eval_result.judge_reasoning,
            error=eval_result.error,
            tests_passed=eval_result.tests_passed,
            tests_total=eval_result.tests_total,
            execution_error=eval_result.execution_error,
            finish_reason=finish_reason,
        )

    async def _process_single_problem(
        self,
        problem: Problem,
        semaphore: asyncio.Semaphore,
        pbar: Optional[atqdm] = None,
    ) -> EvaluationResult:
        from ..utils.prompt_formatter import format_prompt

        test_cases_for_prompt = None
        if self.config.evaluator_type == "code":
            test_cases_for_prompt = problem.metadata.get('test_list', [])

        formatted_prompt = format_prompt(
            problem=problem.problem,
            evaluator_type=self.config.evaluator_type,
            test_cases=test_cases_for_prompt,
        )

        # Fields identifying the problem; identical across every outcome branch.
        problem_fields = dict(
            problem_id=problem.id, question=problem.problem,
            formatted_prompt=formatted_prompt, ground_truth=problem.answer,
        )

        async with semaphore:
            try:
                max_output_tokens = self._calculate_max_output_tokens(formatted_prompt)
                response = await self.client.generate(formatted_prompt, max_output_tokens)

                # Structural check, not string truthiness: an error-shaped
                # response (finish_reason="error") must never reach the
                # evaluator, and the cached error must be non-empty or resume
                # would treat the line as completed and never retry it.
                if response.error is not None or response.finish_reason == "error":
                    error_msg = response.error or "unknown_error: error response with empty message"
                    logger.error(f"Error for problem {problem.id}: {error_msg}")
                    result = EvaluationResult(
                        **problem_fields,
                        model_response=response.text or "", extracted_answer=None, correct=False,
                        tokens=response.tokens, latency=response.latency, error=error_msg,
                        extraction_method="error", finish_reason=response.finish_reason,
                    )
                else:
                    eval_result = await self.evaluator.evaluate(problem, response.text)
                    if eval_result.error:
                        # The model responded but the scorer (e.g. the LLM judge)
                        # failed — record it as an error so the cache re-attempts
                        # on resume, while preserving the real model tokens.
                        logger.error(f"Evaluator error for problem {problem.id}: {eval_result.error}")
                    result = self._result_from_eval(
                        problem_fields, eval_result,
                        model_response=response.text, tokens=response.tokens,
                        latency=response.latency, finish_reason=response.finish_reason,
                    )

                self._append_to_cache(result)
                if pbar:
                    pbar.update(1)
                return result

            except Exception as e:
                logger.error(f"Exception processing problem {problem.id}: {e}")
                result = EvaluationResult(
                    **problem_fields,
                    model_response="", extracted_answer=None, correct=False,
                    tokens=TokenUsage(prompt_tokens=0, answer_tokens=0, reasoning_tokens=0,
                                      output_tokens=0, total_tokens=0),
                    latency=0, error=str(e), extraction_method="exception",
                )
                self._append_to_cache(result)
                if pbar:
                    pbar.update(1)
                return result

    async def _rejudge_cached_problem(
        self,
        problem: Problem,
        cached: EvaluationResult,
        semaphore: asyncio.Semaphore,
        pbar: Optional[atqdm] = None,
    ) -> EvaluationResult:
        async with semaphore:
            try:
                eval_result = await self.evaluator.evaluate(problem, cached.model_response)
                if eval_result.error:
                    logger.error(f"Evaluator error for cached problem {problem.id}: {eval_result.error}")
                result = self._result_from_eval(
                    dict(
                        problem_id=problem.id,
                        question=cached.question or problem.problem,
                        formatted_prompt=cached.formatted_prompt,
                        ground_truth=cached.ground_truth,
                    ),
                    eval_result,
                    model_response=cached.model_response, tokens=cached.tokens,
                    latency=cached.latency, finish_reason=cached.finish_reason,
                )
            except Exception as e:
                logger.error(f"Exception re-judging cached problem {problem.id}: {e}")
                result = cached.model_copy(update={"error": str(e), "extraction_method": "rejudge_exception"})

            self._append_to_cache(result)
            if pbar:
                pbar.update(1)
            return result

    def _append_to_cache(self, result: EvaluationResult) -> None:
        if self.cache is not None:
            self.cache.append(result)

    async def _run_benchmark_async(self) -> List[EvaluationResult]:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        total = len(self.problems) + len(self.rejudge_items)
        pbar = atqdm(
            total=total,
            desc=f"Running {self.config.model} on {total} problems",
            unit="problem",
        )
        tasks = [self._process_single_problem(p, semaphore, pbar) for p in self.problems]
        tasks.extend(
            self._rejudge_cached_problem(p, cached, semaphore, pbar)
            for p, cached in self.rejudge_items
        )
        results = await asyncio.gather(*tasks)
        pbar.close()
        return results

    def run(self) -> ExperimentResult:
        """Run complete benchmark experiment."""
        logger.info("=" * 80)
        logger.info("Starting OckBench Experiment")
        logger.info("=" * 80)
        logger.info(f"Dataset: {self.config.dataset_path}")
        logger.info(f"Model: {self.config.model}")
        logger.info(f"Provider: {self.config.provider}")
        logger.info(f"Evaluator: {self.config.evaluator_type}")
        logger.info(f"Concurrency: {self.config.concurrency}")
        logger.info("=" * 80)

        start_time = time.time()

        logger.info("Loading dataset...")
        loader = get_loader(filepath=self.config.dataset_path)
        self.problems = loader.load()
        logger.info(f"Loaded {len(self.problems)} problems")

        # Build the evaluator early so a missing/invalid config (e.g. math without
        # a judge) fails fast before any model call.
        logger.info("Initializing evaluator...")
        self.evaluator = get_evaluator(self.config.evaluator_type, self.config)

        # Construct the client BEFORE opening the cache. Construction validates
        # the provider name and the per-provider protected-path guard, so an
        # invalid config fails fast without first writing an identity header that
        # would otherwise occupy the cache path and block the corrected rerun.
        logger.info("Initializing API client...")
        self.client = self._create_client()

        try:
            if self.cache_path:
                self.cache = RunCache.open(self.cache_path, self.config)
                completed_ids = self.cache.completed_ids
                # No evaluator-type gate: _is_rejudgable_error is the single
                # arbiter of "recoverable without regeneration", and only
                # judge-backed evaluators produce rows that satisfy it.
                rejudge_map = self.cache.rejudgable_results
                if completed_ids or rejudge_map:
                    remaining = []
                    for p in self.problems:
                        if p.id in completed_ids:
                            continue
                        cached = rejudge_map.get(p.id)
                        if cached is not None:
                            self.rejudge_items.append((p, cached))
                        else:
                            remaining.append(p)
                    self.problems = remaining
                    logger.info(
                        f"Resuming: {len(completed_ids)} cached, "
                        f"{len(self.rejudge_items)} judge-only, {len(self.problems)} remaining"
                    )

            if self.problems or self.rejudge_items:
                logger.info("Running benchmark...")
                new_results = asyncio.run(self._run_and_close_client())
            else:
                logger.info("All problems already cached, no work to do")
                new_results = []
        finally:
            # Constructor/no-work fallback; active async clients are closed
            # inside their event loop by _run_and_close_client.
            if self.client is not None and not self._client_closed:
                self.client.close()

        duration = time.time() - start_time

        # The results file is a pure aggregation of the cache when one is in use,
        # so the two can never diverge; otherwise aggregate the in-memory run.
        if self.cache is not None:
            results, summary = self.cache.aggregate(duration)
        else:
            results = list(new_results)
            summary = summarize(results, duration)

        self._log_summary(summary)

        dataset_name = self.config.dataset_name or Path(self.config.dataset_path).stem
        return ExperimentResult(
            config=self.config, results=results, summary=summary, dataset_name=dataset_name,
        )

    def _log_summary(self, summary: ExperimentSummary) -> None:
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


def run_benchmark(
    config_path: Optional[str] = None,
    output_dir: str = "results",
    log_dir: Optional[str] = None,
    cache: Optional[str] = None,
    **config_overrides,
) -> ExperimentResult:
    """Main entry point for running benchmarks."""
    config_overrides.pop('cache', None)
    config = load_config(config_path, **config_overrides)

    dataset_name = config.dataset_name or Path(config.dataset_path).stem
    from ..utils.logger import get_log_filename

    log_file = Path(log_dir) / get_log_filename(dataset_name, config.model) if log_dir else None
    setup_logger(log_file=str(log_file) if log_file else None)

    runner = BenchmarkRunner(config, cache_path=cache)
    experiment = runner.run()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / get_experiment_filename(experiment.dataset_name, config.model)
    experiment.save_to_file(str(result_file))
    logger.info(f"Results saved to: {result_file}")

    return experiment
