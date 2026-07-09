"""Command-line argument parser for OckBench."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Task presets with default values
TASK_PRESETS = {
    "math": {
        "dataset_path": "data/OckBench_math.jsonl",
        "dataset_name": "OckBench_math",
        "evaluator_type": "math",
    },
    "coding": {
        "dataset_path": "data/OckBench_coding.jsonl",
        "dataset_name": "OckBench_coding",
        "evaluator_type": "code",
        "execution_timeout": 10,
        "include_challenge_tests": True,
    },
    "science": {
        "dataset_path": "data/OckBench_science.jsonl",
        "dataset_name": "OckBench_science",
        "evaluator_type": "science",
    },
}

# Help-epilog examples as (description, argv) pairs. This is the single source of
# truth: the --help text is rendered from it, and a docs regression test
# validates each argv through the real config/inspect path. Every example must be
# a valid current command (math examples include the required judge). Shell-style
# placeholders like $OPENAI_API_KEY are non-empty strings that validate offline.
HELP_EXAMPLES: List[Tuple[str, List[str]]] = [
    ("OpenAI math (the LLM judge is required; its key resolves from JUDGE_API_KEY / OPENAI_API_KEY)",
     ["--model", "gpt-5.2", "--api-key", "$OPENAI_API_KEY", "--base-url", "$OPENAI_BASE_URL",
      "--task", "math", "--max-output-tokens", "128000",
      "--judge-model", "gpt-4o-mini", "--judge-base-url", "$OPENAI_BASE_URL"]),
    ("OpenAI coding (deterministic scorer — no judge needed)",
     ["--model", "gpt-5.2", "--api-key", "$OPENAI_API_KEY", "--base-url", "$OPENAI_BASE_URL",
      "--task", "coding", "--max-output-tokens", "128000"]),
    ("Gemini science (key from GEMINI_API_KEY or --api-key)",
     ["--model", "gemini-3.1-pro-preview", "--provider", "gemini", "--api-key", "$GEMINI_API_KEY",
      "--task", "science", "--max-output-tokens", "65536"]),
    ("Local vLLM/SGLang math (same local server as judge)",
     ["--model", "Qwen/Qwen3-4B", "--api-key", "dummy", "--base-url", "http://localhost:8000/v1",
      "--task", "math", "--max-context-window", "40960",
      "--judge-model", "Qwen/Qwen3-4B", "--judge-base-url", "http://localhost:8000/v1",
      "--judge-api-key", "dummy"]),
    ("Third-party relay, science (OpenAI-compatible proxy)",
     ["--model", "openai/gpt-4o-mini", "--base-url", "https://openrouter.ai/api/v1",
      "--api-key", "$OPENROUTER_API_KEY", "--task", "science", "--max-output-tokens", "32000"]),
    ("Bundled YAML config — inspect the resolved request (no network)",
     ["--config", "configs/openai.yaml", "--api-key", "$OPENAI_API_KEY", "--inspect"]),
]


def _format_help_examples() -> str:
    lines = ["", "Examples:"]
    for description, argv in HELP_EXAMPLES:
        lines.append(f"  # {description}")
        lines.append("  python main.py " + " ".join(argv))
        lines.append("")
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="OckBench - LLM Benchmarking Tool for Reasoning Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_format_help_examples(),
    )

    # Provider: resolved through the provider registry. The four built-ins are
    # chat_completion / openai-responses / anthropic / gemini; an externally
    # registered provider name is accepted too (no fixed choice list).
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="API provider name resolved via the provider registry "
             "(built-ins: chat_completion, openai-responses, anthropic, gemini; default: chat_completion). "
             "When omitted, a provider set in --config is kept.",
    )

    # Task preset
    parser.add_argument(
        "--task",
        type=str,
        choices=["math", "coding", "science"],
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
        default=None,
        help="Model name/identifier (required)",
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
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=None,
        help="Dataset split/subset label recorded in result provenance (e.g. Selected, mini, full)",
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
        help="Maximum output tokens (mutually exclusive with --max-context-window)",
    )
    parser.add_argument(
        "--max-context-window",
        type=int,
        default=None,
        help="Maximum context window (input + output), dynamically calculates output tokens "
             "(mutually exclusive with --max-output-tokens)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter",
    )

    # Request overrides: freely shape the outgoing chat-completions request.
    parser.add_argument(
        "--request-set",
        action="append",
        default=None,
        metavar="PATH=JSON_VALUE",
        help="Set a field at a dotted PATH to a JSON-typed value (repeatable). "
             "The value is parsed as JSON, falling back to a plain string. "
             "Use the ${max_output_tokens} placeholder for the per-problem budget.",
    )
    parser.add_argument(
        "--request-unset",
        action="append",
        default=None,
        metavar="PATH",
        help="Remove the field at a dotted PATH from the request (repeatable).",
    )
    parser.add_argument(
        "--request-set-json",
        action="append",
        default=None,
        metavar="JSON_OBJECT",
        help="Set multiple overrides at once from a JSON object mapping dotted "
             "paths to values (repeatable).",
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
        help="Provider timeout in seconds (streaming providers use this as a per-chunk read timeout)",
    )
    parser.add_argument(
        "--wall-clock-timeout",
        type=int,
        default=None,
        help="Optional wall-clock deadline in seconds for each request attempt. "
             "When exceeded, the attempt is retried through --max-retries.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retries per request",
    )

    # Evaluation configuration. Resolved through the evaluator registry; the
    # built-ins are math / code / science, plus any externally registered task.
    parser.add_argument(
        "--evaluator-type",
        type=str,
        default=None,
        help="Evaluator name resolved via the evaluator registry "
             "(built-ins: math, code, science)",
    )

    # Math LLM judge (the default, required math scorer).
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Math judge model name (required for math scoring)",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="Math judge base URL (OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="Math judge API key (falls back to JUDGE_API_KEY / OPENAI_API_KEY env vars)",
    )

    # Dry-run / inspect: resolve config and print the sanitized request, no network.
    parser.add_argument(
        "--inspect",
        action="store_true",
        default=False,
        help="Resolve config and print the sanitized outgoing request without any network call",
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

    # Resume / caching
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Path to a JSONL cache file for incremental saving and resume. "
             "Results are appended as each problem completes. On restart, "
             "completed problems are skipped automatically.",
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
    parser = create_parser()
    return parser.parse_args(args)


def _parse_cli_request_overrides(
    set_items, unset_items, set_json_items
) -> Tuple[Dict[str, Any], List[str]]:
    """Parse the request-override CLI flags into a (set, unset) pair.

    Raises ValueError on a missing '=', a non-object --request-set-json,
    invalid JSON, or a duplicate set path (no silent last-write-wins).
    """
    set_map: Dict[str, Any] = {}

    def _record(path: str, value: Any) -> None:
        if path in set_map:
            raise ValueError(
                f"duplicate request override path '{path}' on the command line; "
                "each path may be set at most once"
            )
        set_map[path] = value

    for item in set_items or []:
        if "=" not in item:
            raise ValueError(
                f"invalid --request-set '{item}': expected PATH=JSON_VALUE (missing '=')"
            )
        path, raw_value = item.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(f"invalid --request-set '{item}': empty path")
        try:
            value: Any = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        _record(path, value)

    for item in set_json_items or []:
        try:
            parsed = json.loads(item)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --request-set-json '{item}': not valid JSON ({exc})")
        if not isinstance(parsed, dict):
            raise ValueError(
                f"invalid --request-set-json '{item}': expected a JSON object "
                "mapping dotted paths to values"
            )
        for path, value in parsed.items():
            _record(path, value)

    unset_list: List[str] = []
    for item in unset_items or []:
        path = item.strip()
        if not path:
            raise ValueError("invalid --request-unset: empty path")
        if path not in unset_list:
            unset_list.append(path)

    return set_map, unset_list


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build config dict. Priority: CLI args > config file > task preset."""
    config = {}

    # 1. Apply task preset
    task_preset = TASK_PRESETS.get(args.task, {})
    config.update(task_preset)

    # 2. Apply config file if provided
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
        "provider": args.provider,
        "model": args.model,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "max_context_window": args.max_context_window,
        "top_p": args.top_p,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "wall_clock_timeout": args.wall_clock_timeout,
        "max_retries": args.max_retries,
        "evaluator_type": args.evaluator_type,
        "execution_timeout": args.execution_timeout,
        "experiment_name": args.experiment_name,
        "notes": args.notes,
        "cache": args.cache,
    }

    # Handle boolean flags with negation options
    if args.no_include_challenge_tests:
        cli_overrides["include_challenge_tests"] = False
    elif args.include_challenge_tests:
        cli_overrides["include_challenge_tests"] = True

    # Apply non-None overrides
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    # Effective provider default: only when neither --config nor --provider set it,
    # so an explicit provider in a config file is never clobbered by a CLI default.
    config.setdefault("provider", "chat_completion")

    # Assemble the math judge config: YAML base, CLI flags on top.
    judge_cfg = dict(config.get("judge") or {})
    if args.judge_model is not None:
        judge_cfg["model"] = args.judge_model
    if args.judge_base_url is not None:
        judge_cfg["base_url"] = args.judge_base_url
    if args.judge_api_key is not None:
        judge_cfg["api_key"] = args.judge_api_key
    if judge_cfg:
        config["judge"] = judge_cfg

    # Handle mutual exclusivity of max_output_tokens and max_context_window
    # If user explicitly sets one via CLI, remove the other from config
    if args.max_output_tokens is not None:
        config.pop("max_context_window", None)
    elif args.max_context_window is not None:
        config.pop("max_output_tokens", None)

    # Assemble request overrides exactly once: YAML/preset base, CLI on top
    # (CLI wins). Validation and the protected-field guard run here so clients
    # consume the already-merged result without re-parsing.
    yaml_overrides = config.get("request_overrides") or {}
    yaml_set = dict(yaml_overrides.get("set") or {})
    yaml_unset = list(yaml_overrides.get("unset") or [])

    cli_set, cli_unset = _parse_cli_request_overrides(
        args.request_set, args.request_unset, args.request_set_json
    )

    cli_unset_set = set(cli_unset)
    merged_set = {**yaml_set, **cli_set}
    combined_unset = yaml_unset + [path for path in cli_unset if path not in yaml_unset]
    # CLI precedence: a CLI `--request-set PATH=...` overrides a YAML unset of the
    # same path, unless the CLI also explicitly unsets it.
    merged_unset = [
        path for path in combined_unset
        if not (path in cli_set and path not in cli_unset_set)
    ]

    # Standard CLI generation flags take precedence over YAML request overrides
    # that target the same top-level field. If the user passes --temperature or
    # --top-p, drop a YAML set/unset of that path so the CLI flag wins (unless
    # they also gave an explicit --request-* for it on the CLI). --max-output-tokens
    # is intentionally excluded: it controls the budget value, not its placement,
    # which an override may redirect via ${max_output_tokens}.
    cli_override_paths = set(cli_set.keys()) | cli_unset_set
    flag_claimed_paths = set()
    if args.temperature is not None:
        flag_claimed_paths.add("temperature")
    if args.top_p is not None:
        flag_claimed_paths.add("top_p")
    for path in flag_claimed_paths - cli_override_paths:
        merged_set.pop(path, None)
        merged_unset = [entry for entry in merged_unset if entry != path]

    # Empty-path rejection and the protected-field guard run once during
    # BenchmarkConfig validation, so they cover every config-construction path.
    config["request_overrides"] = {"set": merged_set, "unset": merged_unset}

    # Resolve API keys from the environment on the FINAL merged config (provider
    # and judge are now assembled), via the same helper load_config uses — so the
    # CLI path honors the documented JUDGE_API_KEY / OPENAI_API_KEY / provider env
    # fallbacks. Explicit CLI/YAML keys still win (helper only fills when unset).
    from ..core.config import apply_env_keys
    config = apply_env_keys(config)

    return config
