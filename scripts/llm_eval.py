#!/usr/bin/env python3
"""
LLM-based post-hoc evaluator for OckBench results.

This script re-evaluates benchmark results using an LLM judge instead of
regex-based answer extraction. It reads result JSON files and produces
re-scored results with LLM judgments.

Usage:
    python scripts/llm_eval.py results/OckBench_math_gpt-4o_*.json
    python scripts/llm_eval.py results/*.json --output-dir llm_evaluated/
    python scripts/llm_eval.py results/*.json --model gpt-4o --concurrency 10
"""
import argparse
import asyncio
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Default evaluator model
DEFAULT_MODEL = "gpt-4o-mini"

# Evaluation prompt template
EVAL_PROMPT_TEMPLATE = """You are a math problem evaluator. Your task is to determine if a student's answer is correct by comparing it to the ground truth answer.

## Problem
{question}

## Ground Truth Answer
{ground_truth}

## Student's Response
{model_response}

## Instructions
1. Carefully read the student's full response
2. Identify the student's final answer. Look for (in priority order):
   - <answer>...</answer> tags (preferred format)
   - \\boxed{{answer}} format (LaTeX)
   - "The answer is X" or "Final answer: X" statements
   - The last numerical value stated as the conclusion
   - Content after </think> tags (some models use <think>...</think> for reasoning, with the final answer after)
3. Compare the student's final answer to the ground truth
4. The answer is correct if it matches the ground truth numerically or semantically (e.g., "1/2" = "0.5", "two" = "2")
5. Minor formatting differences should be ignored (e.g., "42" vs "42.0", "$100" vs "100")

## Response Format
Respond with ONLY a JSON object in this exact format:
{{"correct": true/false, "extracted_answer": "the student's final answer", "reasoning": "brief explanation of your judgment"}}
"""


@dataclass
class LLMEvalResult:
    """Result from LLM evaluation of a single problem."""
    problem_id: Any
    llm_correct: bool
    llm_extracted_answer: Optional[str]
    llm_reasoning: str
    original_correct: bool
    original_extracted_answer: Optional[Any]
    agreement: bool  # Whether LLM and original evaluator agree
    ground_truth: Any
    error: Optional[str] = None


@dataclass
class LLMEvalSummary:
    """Summary of LLM evaluation results."""
    total_problems: int
    llm_correct_count: int
    llm_accuracy: float
    original_correct_count: int
    original_accuracy: float
    agreement_count: int
    agreement_rate: float
    # OckScore (uses token data from original benchmark run)
    avg_tokens_per_problem: Optional[float] = None
    llm_ock_score: Optional[float] = None
    original_ock_score: Optional[float] = None
    # Disagreement breakdown
    llm_correct_original_wrong: int = 0  # LLM found correct answers that regex missed
    llm_wrong_original_correct: int = 0  # LLM marked wrong but regex said correct
    eval_errors: int = 0


class LLMEvaluator:
    """LLM-based evaluator for math problems."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        concurrency: int = 5,
        timeout: int = 60,
        max_retries: int = 3,
        base_url: Optional[str] = None
    ):
        client_kwargs = {}
        if api_key:
            client_kwargs['api_key'] = api_key
        if base_url:
            client_kwargs['base_url'] = base_url
            if 'api_key' not in client_kwargs:
                client_kwargs['api_key'] = 'dummy-key'
        self.client = AsyncOpenAI(**client_kwargs)
        self.model = model
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_retries = max_retries

    async def evaluate_single(
        self,
        problem_id: Any,
        question: str,
        model_response: str,
        ground_truth: Any,
        original_correct: bool,
        original_extracted: Optional[Any],
        semaphore: asyncio.Semaphore
    ) -> LLMEvalResult:
        """Evaluate a single problem using LLM."""
        async with semaphore:
            # Use last 8000 chars since final answers are always at the end
            truncated_response = model_response[-8000:] if len(model_response) > 8000 else model_response
            prompt = EVAL_PROMPT_TEMPLATE.format(
                question=question,
                ground_truth=ground_truth,
                model_response=truncated_response
            )

            for attempt in range(self.max_retries):
                try:
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                            max_tokens=500,
                        ),
                        timeout=self.timeout
                    )

                    content = response.choices[0].message.content.strip()

                    # Parse JSON response
                    # Handle potential markdown code blocks
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()

                    result = json.loads(content)

                    llm_correct = result.get("correct", False)
                    llm_extracted = result.get("extracted_answer")
                    llm_reasoning = result.get("reasoning", "")

                    return LLMEvalResult(
                        problem_id=problem_id,
                        llm_correct=llm_correct,
                        llm_extracted_answer=str(llm_extracted) if llm_extracted is not None else None,
                        llm_reasoning=llm_reasoning,
                        original_correct=original_correct,
                        original_extracted_answer=original_extracted,
                        agreement=llm_correct == original_correct,
                        ground_truth=ground_truth
                    )

                except json.JSONDecodeError as e:
                    logger.warning(f"Problem {problem_id}: Failed to parse LLM response as JSON (attempt {attempt + 1})")
                    if attempt == self.max_retries - 1:
                        return LLMEvalResult(
                            problem_id=problem_id,
                            llm_correct=False,
                            llm_extracted_answer=None,
                            llm_reasoning="",
                            original_correct=original_correct,
                            original_extracted_answer=original_extracted,
                            agreement=False,
                            ground_truth=ground_truth,
                            error=f"JSON parse error: {e}"
                        )
                except asyncio.TimeoutError:
                    logger.warning(f"Problem {problem_id}: Timeout (attempt {attempt + 1})")
                    if attempt == self.max_retries - 1:
                        return LLMEvalResult(
                            problem_id=problem_id,
                            llm_correct=False,
                            llm_extracted_answer=None,
                            llm_reasoning="",
                            original_correct=original_correct,
                            original_extracted_answer=original_extracted,
                            agreement=False,
                            ground_truth=ground_truth,
                            error="Timeout"
                        )
                except Exception as e:
                    logger.warning(f"Problem {problem_id}: Error {e} (attempt {attempt + 1})")
                    if attempt == self.max_retries - 1:
                        return LLMEvalResult(
                            problem_id=problem_id,
                            llm_correct=False,
                            llm_extracted_answer=None,
                            llm_reasoning="",
                            original_correct=original_correct,
                            original_extracted_answer=original_extracted,
                            agreement=False,
                            ground_truth=ground_truth,
                            error=str(e)
                        )

                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

        # Should not reach here
        return LLMEvalResult(
            problem_id=problem_id,
            llm_correct=False,
            llm_extracted_answer=None,
            llm_reasoning="",
            original_correct=original_correct,
            original_extracted_answer=original_extracted,
            agreement=False,
            ground_truth=ground_truth,
            error="Max retries exceeded"
        )

    async def evaluate_results(self, results: List[Dict[str, Any]]) -> List[LLMEvalResult]:
        """Evaluate all results concurrently."""
        semaphore = asyncio.Semaphore(self.concurrency)

        tasks = [
            self.evaluate_single(
                problem_id=r["problem_id"],
                question=r["question"],
                model_response=r["model_response"],
                ground_truth=r["ground_truth"],
                original_correct=r["correct"],
                original_extracted=r.get("extracted_answer"),
                semaphore=semaphore
            )
            for r in results
        ]

        eval_results = []
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM Evaluation"):
            result = await coro
            eval_results.append(result)

        # Sort by problem_id to maintain order
        eval_results.sort(key=lambda x: x.problem_id)

        return eval_results

    def compute_summary(
        self,
        eval_results: List[LLMEvalResult],
        original_summary: Optional[Dict[str, Any]] = None
    ) -> LLMEvalSummary:
        """Compute summary statistics from evaluation results."""
        total = len(eval_results)
        llm_correct = sum(1 for r in eval_results if r.llm_correct)
        original_correct = sum(1 for r in eval_results if r.original_correct)
        agreement = sum(1 for r in eval_results if r.agreement)
        errors = sum(1 for r in eval_results if r.error)

        # Disagreement breakdown
        llm_correct_original_wrong = sum(
            1 for r in eval_results
            if r.llm_correct and not r.original_correct
        )
        llm_wrong_original_correct = sum(
            1 for r in eval_results
            if not r.llm_correct and r.original_correct
        )

        llm_accuracy = llm_correct / total * 100 if total > 0 else 0
        original_accuracy = original_correct / total * 100 if total > 0 else 0

        # Compute OckScore using token data from original benchmark
        avg_tokens = None
        llm_ock_score = None
        original_ock_score = None
        if original_summary and original_summary.get("avg_tokens_per_problem"):
            avg_tokens = original_summary["avg_tokens_per_problem"]
            token_penalty = 10 * math.log(avg_tokens / 10000 + 1)
            llm_ock_score = llm_accuracy - token_penalty
            original_ock_score = original_accuracy - token_penalty

        return LLMEvalSummary(
            total_problems=total,
            llm_correct_count=llm_correct,
            llm_accuracy=llm_accuracy,
            original_correct_count=original_correct,
            original_accuracy=original_accuracy,
            agreement_count=agreement,
            agreement_rate=agreement / total * 100 if total > 0 else 0,
            avg_tokens_per_problem=avg_tokens,
            llm_ock_score=llm_ock_score,
            original_ock_score=original_ock_score,
            llm_correct_original_wrong=llm_correct_original_wrong,
            llm_wrong_original_correct=llm_wrong_original_correct,
            eval_errors=errors
        )


def load_result_file(filepath: str) -> Dict[str, Any]:
    """Load a result JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_evaluation(
    original_data: Dict[str, Any],
    eval_results: List[LLMEvalResult],
    summary: LLMEvalSummary,
    output_path: str,
    evaluator_model: str
):
    """Save LLM evaluation results to file."""
    output = {
        "original_file": original_data.get("dataset_name", "unknown"),
        "original_model": original_data.get("config", {}).get("model", "unknown"),
        "evaluator_model": evaluator_model,
        "timestamp": datetime.now().isoformat(),
        "summary": asdict(summary),
        "results": [asdict(r) for r in eval_results],
        # Include original summary for comparison
        "original_summary": original_data.get("summary", {})
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def save_rescored_results(
    original_data: Dict[str, Any],
    eval_results: List[LLMEvalResult],
    summary: LLMEvalSummary,
    output_path: str,
    evaluator_model: str
):
    """Save a copy of the original result file with LLM eval accuracy and OckScore written back."""
    import copy
    data = copy.deepcopy(original_data)

    # Build lookup from eval results
    eval_lookup = {r.problem_id: r for r in eval_results}

    # Update per-result correct field with LLM judgment
    for result in data.get("results", []):
        er = eval_lookup.get(result["problem_id"])
        if er and er.error is None:
            result["correct"] = er.llm_correct
            result["extracted_answer"] = er.llm_extracted_answer

    # Update summary with LLM eval scores
    data["summary"]["accuracy"] = summary.llm_accuracy
    data["summary"]["correct_count"] = summary.llm_correct_count
    if summary.llm_ock_score is not None:
        data["summary"]["ock_score"] = summary.llm_ock_score

    # Record evaluator metadata
    data["llm_evaluator"] = {
        "model": evaluator_model,
        "timestamp": datetime.now().isoformat()
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(summary: LLMEvalSummary, original_model: str, evaluator_model: str):
    """Print evaluation summary to console."""
    print("\n" + "=" * 70)
    print("LLM EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Evaluated Model: {original_model}")
    print(f"Evaluator Model: {evaluator_model}")
    print("-" * 70)
    print(f"Total Problems: {summary.total_problems}")
    print(f"\nAccuracy Comparison:")
    print(f"  LLM Evaluator:    {summary.llm_accuracy:6.2f}% ({summary.llm_correct_count}/{summary.total_problems})")
    print(f"  Regex Evaluator:  {summary.original_accuracy:6.2f}% ({summary.original_correct_count}/{summary.total_problems})")
    if summary.llm_ock_score is not None:
        print(f"\nOckScore Comparison (Avg Tokens: {summary.avg_tokens_per_problem:.1f}):")
        print(f"  LLM Evaluator:    {summary.llm_ock_score:6.2f}")
        print(f"  Regex Evaluator:  {summary.original_ock_score:6.2f}")
    print(f"\nAgreement Rate: {summary.agreement_rate:.2f}% ({summary.agreement_count}/{summary.total_problems})")
    print(f"\nDisagreements:")
    print(f"  LLM correct, Regex wrong: {summary.llm_correct_original_wrong}")
    print(f"  LLM wrong, Regex correct: {summary.llm_wrong_original_correct}")
    if summary.eval_errors > 0:
        print(f"\nEvaluation Errors: {summary.eval_errors}")
    print("=" * 70)


def print_disagreements(eval_results: List[LLMEvalResult], original_results: List[Dict], limit: int = 5):
    """Print examples of disagreements between LLM and regex evaluator."""
    disagreements = [
        (er, original_results[i])
        for i, er in enumerate(eval_results)
        if not er.agreement and er.error is None
    ]

    if not disagreements:
        print("\nNo disagreements found!")
        return

    print(f"\n{'=' * 70}")
    print(f"DISAGREEMENT EXAMPLES (showing {min(limit, len(disagreements))} of {len(disagreements)})")
    print("=" * 70)

    for er, orig in disagreements[:limit]:
        print(f"\n--- Problem {er.problem_id} ---")
        print(f"Ground Truth: {er.ground_truth}")
        print(f"Regex: extracted='{er.original_extracted_answer}', correct={er.original_correct}")
        print(f"LLM:   extracted='{er.llm_extracted_answer}', correct={er.llm_correct}")
        print(f"LLM Reasoning: {er.llm_reasoning}")
        print(f"Response snippet: {orig['model_response'][:200]}...")


async def evaluate_file(
    filepath: str,
    evaluator: LLMEvaluator,
    output_dir: str
) -> LLMEvalSummary:
    """Evaluate a single result file."""
    logger.info(f"Loading {filepath}")
    data = load_result_file(filepath)

    results = data.get("results", [])
    if not results:
        logger.warning(f"No results found in {filepath}")
        return None

    original_model = data.get("config", {}).get("model", "unknown")
    logger.info(f"Evaluating {len(results)} problems from model '{original_model}'")

    # Run LLM evaluation
    eval_results = await evaluator.evaluate_results(results)

    # Compute summary (pass original summary for OckScore calculation)
    original_summary = data.get("summary", {})
    summary = evaluator.compute_summary(eval_results, original_summary)

    # Save results
    input_name = Path(filepath).stem
    output_path = Path(output_dir) / f"{input_name}_llm_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_evaluation(data, eval_results, summary, str(output_path), evaluator.model)
    logger.info(f"Saved evaluation to {output_path}")

    # Save a copy of the original result file with LLM eval scores written back
    rescored_path = Path(output_dir) / f"{input_name}_llm_rescored.json"
    save_rescored_results(data, eval_results, summary, str(rescored_path), evaluator.model)
    logger.info(f"Saved rescored results to {rescored_path}")

    # Print summary
    print_summary(summary, original_model, evaluator.model)
    print_disagreements(eval_results, results)

    return summary


async def main():
    parser = argparse.ArgumentParser(
        description="LLM-based post-hoc evaluator for OckBench results"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Result JSON files to evaluate"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: from OPENAI_API_KEY or GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Evaluator model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent API requests (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/llm_eval",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (e.g., http://localhost:8000/v1 for local servers, "
             "or https://generativelanguage.googleapis.com/v1beta/openai/ for Gemini)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)"
    )

    args = parser.parse_args()

    # Resolve API key: --api-key > OPENAI_API_KEY > GEMINI_API_KEY > dummy (for local)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.base_url:
        print("Error: API key required. Set OPENAI_API_KEY/GEMINI_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    # Create evaluator
    evaluator = LLMEvaluator(
        api_key=api_key,
        model=args.model,
        concurrency=args.concurrency,
        timeout=args.timeout,
        base_url=args.base_url
    )

    # Process each file
    all_summaries = []
    for filepath in args.files:
        if not Path(filepath).exists():
            logger.warning(f"File not found: {filepath}")
            continue

        summary = await evaluate_file(filepath, evaluator, args.output_dir)
        if summary:
            all_summaries.append((filepath, summary))

    # Print overall summary if multiple files
    if len(all_summaries) > 1:
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY (All Files)")
        print("=" * 70)
        for filepath, summary in all_summaries:
            name = Path(filepath).stem
            print(f"{name}: LLM={summary.llm_accuracy:.1f}% Regex={summary.original_accuracy:.1f}% Agreement={summary.agreement_rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
