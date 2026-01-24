#!/usr/bin/env python3
"""
Convert external lm-evaluation-harness results to OckBench format.

Processes Qwen3 model results (excluding *-Thinking-2507 model variants),
treating enable_thinking=true and enable_thinking=false as separate model variants.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.math_eval import MathEvaluator

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, using character-based estimation")


# Model directories to process (excluding *-Thinking-2507 variants)
MODEL_DIRS = [
    "Qwen__Qwen3-4B",
    "Qwen__Qwen3-8B",
    "Qwen__Qwen3-14B",
    "Qwen__Qwen3-30B-A3B",
    "Qwen__Qwen3-32B",
    "Qwen__Qwen3-235B-A22B",
]

# Dataset mapping: external name -> (OckBench name, evaluator_type)
DATASET_CONFIG = {
    "gsm8k": ("GSM8K", "math"),
    "aime24": ("AIME24", "math"),
    "aime25": ("AIME25", "math"),
    "mbpp": ("MBPP", "code"),  # Skip for now as code evaluation is complex
}

# Only process math datasets for now
DATASETS_TO_PROCESS = ["gsm8k", "aime24", "aime25"]


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for text."""
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            pass
    # Fallback: rough estimation (1 token ≈ 4 chars for English)
    return len(text) // 4


def extract_thinking_and_answer(response: str) -> Tuple[str, str, int, int]:
    """
    Extract thinking content and final answer from response.
    Returns: (thinking_content, answer_content, thinking_tokens, answer_tokens)
    """
    # Check for <think>...</think> tags
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response, re.DOTALL)

    if think_match:
        thinking_content = think_match.group(1).strip()
        # Answer is everything after </think>
        answer_start = think_match.end()
        answer_content = response[answer_start:].strip()
    else:
        thinking_content = ""
        answer_content = response

    thinking_tokens = estimate_tokens(thinking_content)
    answer_tokens = estimate_tokens(answer_content)

    return thinking_content, answer_content, thinking_tokens, answer_tokens


def parse_external_sample(sample: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Parse external sample format to extract relevant fields."""
    doc = sample.get("doc", {})

    # Extract question based on dataset format
    if dataset == "gsm8k":
        question = doc.get("question", "")
        ground_truth = doc.get("answer", "")
        # Extract numeric answer from GSM8K format (#### NUMBER)
        if "####" in str(ground_truth):
            ground_truth = str(ground_truth).split("####")[-1].strip()
    elif dataset in ["aime24", "aime25"]:
        question = doc.get("Problem", doc.get("problem", ""))
        ground_truth = doc.get("Answer", doc.get("answer", ""))
    else:
        question = doc.get("question", doc.get("Problem", doc.get("problem", "")))
        ground_truth = doc.get("answer", doc.get("Answer", ""))

    # Extract model response
    resps = sample.get("resps", [[]])
    model_response = resps[0][0] if resps and resps[0] else ""

    # Extract prompt from arguments
    args = sample.get("arguments", {})
    gen_args = args.get("gen_args_0", {})
    prompt_arg = gen_args.get("arg_0", [""])
    if prompt_arg and isinstance(prompt_arg[0], str):
        try:
            prompt_data = json.loads(prompt_arg[0])
            if isinstance(prompt_data, list) and prompt_data:
                formatted_prompt = prompt_data[0].get("content", "")
            else:
                formatted_prompt = prompt_arg[0]
        except json.JSONDecodeError:
            formatted_prompt = prompt_arg[0]
    else:
        formatted_prompt = str(prompt_arg)

    return {
        "doc_id": sample.get("doc_id", 0),
        "question": question,
        "ground_truth": ground_truth,
        "model_response": model_response,
        "formatted_prompt": formatted_prompt,
    }


def convert_sample_to_ockbench(
    parsed: Dict[str, Any],
    dataset_name: str,
    evaluator: MathEvaluator,
) -> Dict[str, Any]:
    """Convert parsed sample to OckBench EvaluationResult format."""

    # Extract thinking and answer parts
    thinking, answer, thinking_tokens, answer_tokens = extract_thinking_and_answer(
        parsed["model_response"]
    )

    # Estimate prompt tokens
    prompt_tokens = estimate_tokens(parsed["formatted_prompt"])

    # Evaluate correctness using OckBench evaluator
    extracted_answer, extraction_method = evaluator.extract_answer(parsed["model_response"])
    correct = evaluator.compare_answers(extracted_answer, parsed["ground_truth"])

    # Build problem ID
    problem_id = f"{dataset_name}-{parsed['doc_id']}"

    # Build token usage
    output_tokens = thinking_tokens + answer_tokens
    tokens = {
        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
        "reasoning_tokens": thinking_tokens,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }

    return {
        "problem_id": problem_id,
        "question": parsed["question"],
        "formatted_prompt": parsed["formatted_prompt"],
        "ground_truth": parsed["ground_truth"],
        "model_response": parsed["model_response"],
        "extracted_answer": extracted_answer,
        "correct": correct,
        "tokens": tokens,
        "latency": 0.0,  # Not available in external format
        "error": None,
        "extraction_method": extraction_method,
        "tests_passed": None,
        "tests_total": None,
        "execution_error": None,
    }


def compute_summary(results: List[Dict[str, Any]], duration: float = 0.0) -> Dict[str, Any]:
    """Compute summary statistics from results."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    total_prompt = sum(r["tokens"]["prompt_tokens"] for r in results)
    total_answer = sum(r["tokens"]["answer_tokens"] for r in results)
    total_reasoning = sum(r["tokens"]["reasoning_tokens"] for r in results)
    total_output = sum(r["tokens"]["output_tokens"] for r in results)
    total_tokens = sum(r["tokens"]["total_tokens"] for r in results)

    errors = sum(1 for r in results if r.get("error"))

    return {
        "total_problems": total,
        "correct_count": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0.0,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt,
        "total_answer_tokens": total_answer,
        "total_reasoning_tokens": total_reasoning,
        "total_output_tokens": total_output,
        "avg_tokens_per_problem": total_tokens / total if total > 0 else 0.0,
        "avg_latency": 0.0,  # Not available
        "total_duration": duration,
        "error_count": errors,
    }


def build_config(
    model_name: str,
    dataset_name: str,
    dataset_path: str,
    enable_thinking: bool,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Build OckBench config from external metadata."""
    return {
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "provider": "generic",
        "model": model_name,
        "base_url": metadata.get("base_url", "http://localhost:8000/v1"),
        "api_key": None,
        "temperature": metadata.get("temperature", 0.0),
        "max_output_tokens": metadata.get("max_gen_toks", 30000),
        "max_context_window": None,
        "reasoning_effort": None,
        "top_p": None,
        "enable_thinking": enable_thinking,
        "concurrency": metadata.get("num_concurrent", 32),
        "timeout": 600,
        "max_retries": 3,
        "evaluator_type": "math",
        "enforce_output_format": False,
        "custom_format_instruction": None,
        "execution_timeout": 5,
        "include_challenge_tests": True,
        "experiment_name": None,
        "notes": f"Converted from external lm-evaluation-harness results",
    }


def get_timestamp_thinking_map(model_dir: Path) -> Dict[str, bool]:
    """Get mapping of timestamps to enable_thinking value."""
    mapping = {}
    for results_file in model_dir.glob("results_*.json"):
        timestamp = results_file.stem.replace("results_", "")
        with open(results_file, "r") as f:
            data = json.load(f)
        # Get enable_thinking from any task config
        configs = data.get("configs", {})
        for task_config in configs.values():
            metadata = task_config.get("metadata", {})
            if "enable_thinking" in metadata:
                mapping[timestamp] = metadata["enable_thinking"]
                break
    return mapping


def process_model_directory(
    model_dir: Path,
    output_dir: Path,
    datasets: List[str],
) -> List[str]:
    """Process all samples in a model directory."""
    created_files = []

    # Get timestamp -> thinking mapping
    ts_thinking_map = get_timestamp_thinking_map(model_dir)

    if not ts_thinking_map:
        print(f"  Warning: No timestamp mapping found, skipping")
        return created_files

    # Extract model name from directory (e.g., "Qwen__Qwen3-14B" -> "Qwen/Qwen3-14B")
    dir_name = model_dir.name
    base_model = dir_name.replace("__", "/")

    # Initialize evaluator
    evaluator = MathEvaluator()

    for dataset in datasets:
        ockbench_name, eval_type = DATASET_CONFIG.get(dataset, (dataset.upper(), "math"))

        for timestamp, enable_thinking in ts_thinking_map.items():
            sample_file = model_dir / f"samples_{dataset}_{timestamp}.jsonl"

            if not sample_file.exists():
                print(f"  Warning: {sample_file.name} not found, skipping")
                continue

            # Determine model variant name
            variant = "Thinking" if enable_thinking else "Instruct"
            model_name = f"{base_model}-{variant}"

            print(f"  Processing {dataset} ({variant})...")

            # Read and convert samples
            results = []
            with open(sample_file, "r") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        parsed = parse_external_sample(sample, dataset)
                        converted = convert_sample_to_ockbench(
                            parsed, ockbench_name, evaluator
                        )
                        results.append(converted)

            if not results:
                print(f"    No samples found, skipping")
                continue

            # Load metadata from results file
            results_file = model_dir / f"results_{timestamp}.json"
            with open(results_file, "r") as f:
                results_data = json.load(f)

            configs = results_data.get("configs", {})
            task_config = configs.get(dataset, {})
            metadata = task_config.get("metadata", {})

            # Compute summary
            summary = compute_summary(results)

            # Build config
            config = build_config(
                model_name=model_name,
                dataset_name=ockbench_name,
                dataset_path=f"data/{ockbench_name}.jsonl",
                enable_thinking=enable_thinking,
                metadata=metadata,
            )

            # Build experiment result
            # Convert timestamp format for output filename
            ts_clean = timestamp.replace("T", "_").replace("-", "").replace(":", "")[:15]

            experiment = {
                "config": config,
                "results": results,
                "summary": summary,
                "timestamp": timestamp.replace("T", " ").replace("-", ":"),
                "dataset_name": ockbench_name,
            }

            # Save to file
            # Model name for filename: "Qwen/Qwen3-14B-Thinking" -> "Qwen_Qwen3-14B-Thinking"
            model_fname = model_name.replace("/", "_")
            output_file = output_dir / f"{ockbench_name}_{model_fname}_{ts_clean}.json"

            with open(output_file, "w") as f:
                json.dump(experiment, f, indent=2, ensure_ascii=False)

            created_files.append(str(output_file))
            print(f"    Created: {output_file.name}")
            print(f"    Accuracy: {summary['accuracy']:.2f}% ({summary['correct_count']}/{summary['total_problems']})")

    return created_files


def main():
    """Main entry point."""
    # Paths
    external_dir = Path("/home/junxiong/zdu/OckBench/external/results")
    output_dir = Path("/home/junxiong/zdu/OckBench/results/converted")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Converting external results to OckBench format")
    print(f"Output directory: {output_dir}")
    print(f"Datasets: {DATASETS_TO_PROCESS}")
    print(f"Models: {MODEL_DIRS}")
    print()

    all_created = []

    for model_dir_name in MODEL_DIRS:
        model_dir = external_dir / model_dir_name

        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist, skipping")
            continue

        print(f"Processing {model_dir_name}...")
        created = process_model_directory(model_dir, output_dir, DATASETS_TO_PROCESS)
        all_created.extend(created)
        print()

    print(f"Conversion complete! Created {len(all_created)} files.")
    return all_created


if __name__ == "__main__":
    main()
