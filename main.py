#!/usr/bin/env python3
"""
OckBench - LLM Benchmarking Tool for Reasoning Tasks

Main CLI entry point for running benchmarks.
"""
import sys
import logging

from src.core.runner import run_benchmark
from src.utils.parser import parse_args, build_config

logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for running benchmarks."""
    args = parse_args()

    try:
        config = build_config(args)
        cache = config.pop('cache', None)
        experiment = run_benchmark(
            config_path=None,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            cache=cache,
            **config
        )

        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Dataset: {experiment.dataset_name}")
        print(f"Model: {experiment.config.model}")
        print(f"Accuracy: {experiment.summary.accuracy:.2f}% "
              f"({experiment.summary.correct_count}/{experiment.summary.total_problems})")
        print(f"Total Tokens: {experiment.summary.total_tokens:,}")
        print(f"  Prompt: {experiment.summary.total_prompt_tokens:,}")
        print(f"  Answer: {experiment.summary.total_answer_tokens:,}")
        print(f"  Reasoning: {experiment.summary.total_reasoning_tokens:,}")
        print(f"  Output: {experiment.summary.total_output_tokens:,}")
        print(f"Avg Tokens/Problem: {experiment.summary.avg_tokens_per_problem:.1f}")
        print(f"Duration: {experiment.summary.total_duration:.2f}s")
        print("=" * 80)

        # Exit with error code if there were errors
        if experiment.summary.error_count > 0:
            print(f"\nWarning: {experiment.summary.error_count} problems had errors")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return 130

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
