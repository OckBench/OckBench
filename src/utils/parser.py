"""
Command-line argument parser for OckBench.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any


# Provider presets with default values
PROVIDER_PRESETS = {
    "openai": {
        "provider": "openai",
        "concurrency": 20,
        "timeout": 120,
        "max_retries": 3,
        "max_output_tokens": 4096,
    },
    "gemini": {
        "provider": "gemini",
        "concurrency": 5,
        "timeout": 120,
        "max_retries": 3,
        "temperature": 0.0,
        "max_output_tokens": 8192,
    },
    "generic": {
        "provider": "generic",
        "base_url": "http://localhost:8000/v1",
        "concurrency": 20,
        "timeout": 600,
        "max_retries": 3,
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "max_context_window": 40960,
    },
}

# Task presets with default values
TASK_PRESETS = {
    "math": {
        "dataset_path": "data/OckBench_math.jsonl",
        "dataset_name": "OckBench_math",
        "evaluator_type": "math",
        "enforce_output_format": True,
    },
    "coding": {
        "dataset_path": "data/OckBench_coding.jsonl",
        "dataset_name": "OckBench_coding",
        "evaluator_type": "code",
        "enforce_output_format": True,
        "execution_timeout": 10,
        "include_challenge_tests": True,
    },
}


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="OckBench - LLM Benchmarking Tool for Reasoning Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with OpenAI GPT-4o on math tasks
  python main.py --model gpt-4o --provider openai --task math

  # Run with Gemini on coding tasks
  python main.py --model gemini-2.5-pro --provider gemini --task coding

  # Run with local model via vLLM
  python main.py --model qwen3-4b --provider generic --base-url http://localhost:8000/v1

  # Run with custom dataset
  python main.py --model gpt-4o --dataset-path data/custom.jsonl --evaluator-type math

  # Load from config file with overrides
  python main.py --config config.yaml --model gpt-4o-mini
        """,
    )

    # Provider preset
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "gemini", "generic"],
        default="openai",
        help="API provider type (default: openai)",
    )

    # Task preset
    parser.add_argument(
        "--task",
        type=str,
        choices=["math", "coding"],
        default="math",
        help="Task type preset (default: math)",
    )

    # Config file (optional)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional, CLI args override config file)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name/identifier (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (can also use env vars: OPENAI_API_KEY, GEMINI_API_KEY, API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for API (for generic/local providers)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file (overrides task preset)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name of the dataset for logging",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output tokens",
    )
    parser.add_argument(
        "--max-context-window",
        type=int,
        default=None,
        help="Maximum context window (input + output). If set, max_output_tokens calculated dynamically",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Reasoning effort level (for o1/o3 models)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--enable-thinking",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=None,
        metavar="true|false",
        help="Enable/disable thinking mode for Qwen3 models",
    )

    # Runtime configuration
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent API requests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retries per request",
    )

    # Evaluation configuration
    parser.add_argument(
        "--evaluator-type",
        type=str,
        choices=["math", "code"],
        default=None,
        help="Type of evaluator to use",
    )
    parser.add_argument(
        "--enforce-output-format",
        action="store_true",
        default=None,
        help="Add instructions to prompt to enforce consistent output format",
    )
    parser.add_argument(
        "--no-enforce-output-format",
        action="store_true",
        help="Disable output format enforcement",
    )
    parser.add_argument(
        "--custom-format-instruction",
        type=str,
        default=None,
        help="Custom format instruction (overrides default)",
    )

    # Code evaluation specific
    parser.add_argument(
        "--execution-timeout",
        type=int,
        default=None,
        help="Timeout for code execution in seconds",
    )
    parser.add_argument(
        "--include-challenge-tests",
        action="store_true",
        default=None,
        help="Include challenge tests in code evaluation",
    )
    parser.add_argument(
        "--no-include-challenge-tests",
        action="store_true",
        help="Exclude challenge tests from code evaluation",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for output files (default: results)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for log files",
    )

    # Metadata
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Additional notes about the experiment",
    )

    return parser


def parse_args(args=None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = create_parser()
    return parser.parse_args(args)


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build configuration dictionary from args, applying presets and overrides.

    Priority (highest to lowest):
    1. CLI arguments
    2. Config file
    3. Task preset
    4. Provider preset

    Args:
        args: Parsed argument namespace.

    Returns:
        Configuration dictionary ready to pass to run_benchmark.
    """
    config = {}

    # 1. Apply provider preset
    provider_preset = PROVIDER_PRESETS.get(args.provider, {})
    config.update(provider_preset)

    # 2. Apply task preset
    task_preset = TASK_PRESETS.get(args.task, {})
    config.update(task_preset)

    # 3. Apply config file if provided
    if args.config:
        import yaml
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
                config.update(file_config)
        else:
            print(f"Warning: Config file not found: {args.config}", file=sys.stderr)

    # 4. Apply CLI overrides (only non-None values)
    cli_overrides = {
        "model": args.model,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_context_window": args.max_context_window,
        "reasoning_effort": args.reasoning_effort,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "evaluator_type": args.evaluator_type,
        "custom_format_instruction": args.custom_format_instruction,
        "execution_timeout": args.execution_timeout,
        "experiment_name": args.experiment_name,
        "notes": args.notes,
    }

    # Handle boolean flags with negation options
    if args.no_enforce_output_format:
        cli_overrides["enforce_output_format"] = False
    elif args.enforce_output_format:
        cli_overrides["enforce_output_format"] = True

    if args.no_include_challenge_tests:
        cli_overrides["include_challenge_tests"] = False
    elif args.include_challenge_tests:
        cli_overrides["include_challenge_tests"] = True

    # Apply non-None overrides
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    return config
