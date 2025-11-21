#!/usr/bin/env python3
"""
OckBench - LLM Benchmarking Tool for Reasoning Tasks

Main CLI entry point for running benchmarks.
"""
import argparse
import sys
from pathlib import Path

from src.core.runner import run_benchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OckBench: Benchmark LLMs on reasoning tasks with token counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python main.py --config configs/aime25.yaml
  
  # Run with inline parameters
  python main.py --dataset data/GSM8K.jsonl --provider openai --model gpt-4 --api-key sk-xxx
  
  # Run with environment variable for API key
  export OPENAI_API_KEY=sk-xxx
  python main.py --dataset data/GSM8K.jsonl --provider openai --model gpt-4
  
  # Run with custom concurrency and temperature
  python main.py --config configs/gsm8k.yaml --concurrency 10 --temperature 0.7
        """
    )
    
    # Config file or inline parameters
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset file (JSONL format)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Name of dataset for logging (default: inferred from filename)'
    )
    
    # Model parameters
    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'gemini', 'generic'],
        help='API provider (openai, gemini, or generic for OpenAI-compatible)'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name/identifier'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        help='Base URL for API (for generic/local providers)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key (or use environment variables: OPENAI_API_KEY, GEMINI_API_KEY)'
    )
    
    # Generation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature (default: 0.0)'
    )
    parser.add_argument(
        '--max-output-tokens',
        type=int,
        help='Maximum output tokens (default: 4096, can be omitted if max-context-window is set)'
    )
    parser.add_argument(
        '--max-context-window',
        type=int,
        help='Maximum context window (input + output). If set, max-output-tokens will be calculated dynamically per problem'
    )
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        help='Reasoning effort level for o1/o3 models (low, medium, high)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        help='Nucleus sampling parameter'
    )
    
    # Runtime parameters
    parser.add_argument(
        '--concurrency',
        type=int,
        help='Number of concurrent API requests (default: 5)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        help='Request timeout in seconds (default: 120)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        help='Maximum retry attempts (default: 3)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--evaluator',
        type=str,
        default='math',
        help='Evaluator type (default: math)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        help='Directory to save logs (default: no log file)'
    )
    
    # Metadata
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Custom experiment name'
    )
    parser.add_argument(
        '--notes',
        type=str,
        help='Additional notes about the experiment'
    )
    
    return parser.parse_args()


def validate_args(args):
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments
    
    Raises:
        ValueError: If arguments are invalid
    """
    # Must have either config file or required inline parameters
    if not args.config:
        required = ['dataset', 'provider', 'model']
        missing = [arg for arg in required if not getattr(args, arg.replace('-', '_'))]
        
        if missing:
            raise ValueError(
                f"When not using --config, you must provide: {', '.join(['--' + m for m in missing])}"
            )
        
        # Check dataset file exists
        if not Path(args.dataset).exists():
            raise ValueError(f"Dataset file not found: {args.dataset}")
        
        # Generic provider requires base_url
        if args.provider == 'generic' and not args.base_url:
            raise ValueError("--base-url is required when using --provider generic")


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Validate arguments
        validate_args(args)
        
        # Build config overrides from CLI args
        config_overrides = {}
        
        # Map CLI args to config fields
        arg_mapping = {
            'dataset': 'dataset_path',
            'dataset_name': 'dataset_name',
            'provider': 'provider',
            'model': 'model',
            'base_url': 'base_url',
            'api_key': 'api_key',
            'temperature': 'temperature',
            'max_output_tokens': 'max_output_tokens',
            'max_context_window': 'max_context_window',
            'reasoning_effort': 'reasoning_effort',
            'top_p': 'top_p',
            'concurrency': 'concurrency',
            'timeout': 'timeout',
            'max_retries': 'max_retries',
            'evaluator': 'evaluator_type',
            'experiment_name': 'experiment_name',
            'notes': 'notes',
        }
        
        for arg_name, config_field in arg_mapping.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                config_overrides[config_field] = value
        
        # Run benchmark
        experiment = run_benchmark(
            config_path=args.config,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            **config_overrides
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
        print(f"  Completion: {experiment.summary.total_completion_tokens:,}")
        print(f"  Reasoning: {experiment.summary.total_reasoning_tokens:,}")
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
