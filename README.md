# OckBench

**Efficiency-Aware LLM Benchmark**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

OckBench measures both accuracy and token efficiency for LLM reasoning tasks. It tracks prompt tokens, output tokens, and reasoning tokens separately to help evaluate the efficiency-accuracy trade-off.

## Features

- **Dual-axis evaluation**: Tracks accuracy (Pass@1) and token usage (prompt, output, reasoning)
- **Multiple providers**: OpenAI, Gemini, and generic OpenAI-compatible endpoints (vLLM, SGLang, LMDeploy)
- **Math and code tasks**: Robust answer extraction with 10+ regex patterns for math, execution-based evaluation for code
- **Dynamic token management**: Calculates max_output_tokens based on model context windows

## Installation

```bash
git clone https://github.com/OckBench/OckBench.git
cd OckBench
pip install -r requirements.txt
```

## Usage

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
python main.py --model gpt-4o --task math
```

### Local Models (vLLM/SGLang)

```bash
python main.py \
    --provider generic \
    --model Qwen/Qwen3-4B \
    --base-url http://localhost:8000/v1 \
    --task coding
```

## Configuration

All parameters can be set via command line. Use `python main.py --help` for full options.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name | gpt-4o-mini |
| `--provider` | API provider (openai/gemini/generic) | openai |
| `--task` | Task type (math/coding) | math |
| `--concurrency` | Parallel requests | 20 |
| `--temperature` | Sampling temperature | 0.0 |
| `--output-dir` | Results directory | results |

## Output

Results are saved to `results/` as JSON:

```json
{
  "summary": {
    "accuracy": 85.5,
    "total_tokens": 125000,
    "avg_tokens_per_problem": 125.0
  },
  "results": [
    {
      "problem_id": 1,
      "correct": true,
      "tokens": {
        "prompt_tokens": 50,
        "reasoning_tokens": 120,
        "output_tokens": 125
      }
    }
  ]
}
```

## Dataset Format

JSONL format with one problem per line:

```json
{"id": 1, "problem": "What is 2+2?", "answer": "4"}
```

For code tasks, include test cases in metadata:

```json
{"id": 1, "problem": "Write a function...", "answer": "def func()...", "metadata": {"test_cases": ["assert func(1) == 2"]}}
```
