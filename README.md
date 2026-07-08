<!-- markdownlint-disable MD001 MD013 MD041 -->
<h1 align="center">OckBench</h1>

<h3 align="center">
Efficiency-aware benchmark for LLM reasoning.
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2511.05722"><b>Paper</b></a> |
  <a href="https://ockbench.github.io/"><b>Website</b></a> |
  <a href="https://huggingface.co/ockbench"><b>HuggingFace</b></a>
</p>

---

## Overview

OckBench evaluates LLM reasoning with two signals:

- **Accuracy**: whether the model solves the task.
- **Token cost**: how many prompt, answer, and reasoning tokens it spends.

The benchmark reports both raw accuracy and **OckScore**, a single score that
penalizes unnecessary token use:

```text
OckScore = accuracy - 10 * log(avg_tokens_per_problem / 10000 + 1)
```

OckBench includes three task families:

| Task | Dataset preset | Evaluator |
| --- | --- | --- |
| Math | `data/OckBench_math.jsonl` | Extracts the model answer, then scores it with a required LLM judge |
| Coding | `data/OckBench_coding.jsonl` | Executes generated code against tests |
| Science | `data/OckBench_science.jsonl` | Extracts and checks the final multiple-choice answer |

It supports OpenAI-compatible chat completions, OpenAI Responses API, Anthropic,
Gemini, local vLLM/SGLang servers, and third-party OpenAI-compatible relays.

## Highlights

- Efficiency-aware scoring with prompt, answer, reasoning, output, and total
  token accounting.
- First-class support for reasoning/thinking controls through uniform request
  overrides instead of provider-specific flags.
- Incremental JSONL cache for resumable benchmark runs.
- Secret-masked provenance in result files and `--inspect` output.
- Registry-based providers and evaluators for custom integrations.
- YAML configs for reproducible runs, with CLI flags available for quick
  experiments.

## Installation

OckBench requires Python 3.12 or newer.

```bash
git clone https://github.com/OckBench/OckBench.git
cd OckBench

uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -e .
```

For development and tests:

```bash
uv sync --dev
```

## Quick Start

Every run must specify exactly one output budget mode:

- `--max-output-tokens`: fixed output-token budget per problem.
- `--max-context-window`: total context window; OckBench estimates prompt
  tokens and uses the remaining budget for output.

### OpenAI-Compatible Chat Completions

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Math requires an LLM judge:

```bash
python main.py \
  --model gpt-5.2 \
  --api-key "$OPENAI_API_KEY" \
  --base-url "$OPENAI_BASE_URL" \
  --task math \
  --max-output-tokens 128000 \
  --judge-model gpt-4o-mini \
  --judge-base-url "$OPENAI_BASE_URL"
```

Coding and science use deterministic evaluators, so no judge is needed:

```bash
python main.py \
  --model gpt-5.2 \
  --api-key "$OPENAI_API_KEY" \
  --base-url "$OPENAI_BASE_URL" \
  --task coding \
  --max-output-tokens 128000

python main.py \
  --model gpt-5.2 \
  --api-key "$OPENAI_API_KEY" \
  --base-url "$OPENAI_BASE_URL" \
  --task science \
  --max-output-tokens 128000
```

### Local vLLM or SGLang Server

Start an OpenAI-compatible local server first:

```bash
vllm serve Qwen/Qwen3-4B --port 8000
```

Then run the benchmark:

```bash
python main.py \
  --model Qwen/Qwen3-4B \
  --api-key dummy \
  --base-url http://localhost:8000/v1 \
  --task math \
  --max-context-window 40960 \
  --judge-model Qwen/Qwen3-4B \
  --judge-base-url http://localhost:8000/v1 \
  --judge-api-key dummy
```

For coding or science:

```bash
python main.py \
  --model Qwen/Qwen3-4B \
  --api-key dummy \
  --base-url http://localhost:8000/v1 \
  --task science \
  --max-context-window 40960
```

### YAML Configs

Use bundled configs when you want reproducible provider settings:

```bash
python main.py --config configs/openai.yaml --api-key "$OPENAI_API_KEY"
```

CLI flags override YAML fields. The task preset is applied first, then YAML,
then CLI overrides.

Bundled configs:

| Config | Provider | Intended use |
| --- | --- | --- |
| `configs/openai.yaml` | `chat_completion` | OpenAI-compatible chat completions |
| `configs/openai_responses.yaml` | `openai-responses` | OpenAI `/v1/responses` endpoint |
| `configs/anthropic.yaml` | `anthropic` | Anthropic Messages API |
| `configs/gemini.yaml` | `gemini` | Gemini API |
| `configs/local.yaml` | `chat_completion` | Local vLLM/SGLang server |
| `configs/relay.yaml` | `chat_completion` | Third-party OpenAI-compatible relay |

## Configuration Reference

### Task Presets

`--task` selects a default dataset and evaluator:

| `--task` | Default dataset | Default evaluator |
| --- | --- | --- |
| `math` | `data/OckBench_math.jsonl` | `math` |
| `coding` | `data/OckBench_coding.jsonl` | `code` |
| `science` | `data/OckBench_science.jsonl` | `science` |

You can override the dataset with `--dataset-path`, `--dataset-name`, and
`--dataset-split`. Selected and mini splits are also included under `data/`.

### Provider Keys

`chat_completion` requires `api_key` and `base_url` via CLI or YAML because it
is used for OpenAI, local servers, and relays. Other built-in providers can
resolve keys from environment variables:

| Provider | Environment fallback |
| --- | --- |
| `openai-responses` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| Math judge | `JUDGE_API_KEY`, then `OPENAI_API_KEY` |

### Request Overrides

Providers place reasoning, thinking, sampling, and token-budget knobs in
different request fields. OckBench exposes one request-shaping mechanism for all
providers:

- `--request-set PATH=JSON_VALUE`: set a dotted request field. Values are parsed
  as JSON first, then as strings.
- `--request-unset PATH`: remove a dotted request field.
- `--request-set-json '{"path": value}'`: set several fields at once.

`${max_output_tokens}` is replaced at request-build time with the per-problem
output budget, so it works with both fixed and context-window budgeting.

Examples:

```bash
# Qwen thinking through vLLM/SGLang
python main.py ... \
  --request-set extra_body.chat_template_kwargs.enable_thinking=true
```

```yaml
# OpenAI-compatible reasoning model on chat completions
request_overrides:
  set:
    max_completion_tokens: ${max_output_tokens}
    reasoning_effort: high
  unset:
    - max_tokens
    - temperature
    - top_p
```

```yaml
# OpenAI Responses API
provider: openai-responses
request_overrides:
  set:
    reasoning.effort: high
  unset:
    - temperature
```

```yaml
# Gemini thinking budget
provider: gemini
request_overrides:
  set:
    config.thinking_config.thinking_budget: 8192
```

```yaml
# Anthropic effort hint
provider: anthropic
request_overrides:
  set:
    output_config.effort: high
```

Overrides cannot replace provider-protected fields such as `model`, `messages`,
`input`, `stream`, or stream options; invalid overrides fail before the run
starts.

### Inspect a Resolved Request

Use `--inspect` to resolve the final config and print the exact sanitized
request without making a network call:

```bash
python main.py --config configs/openai.yaml --api-key "$OPENAI_API_KEY" --inspect
```

API keys, bearer tokens, secret headers, credentials embedded in URLs, and judge
credentials are masked.

### Removed Reasoning Flags

The older `--enable-thinking` and `--reasoning-effort` flags were removed.
Express those settings through `request_overrides` instead:

| Former behavior | Current override |
| --- | --- |
| Local Qwen thinking toggle | `--request-set extra_body.chat_template_kwargs.enable_thinking=true` |
| DeepSeek thinking, no sampling params | `--request-set extra_body.thinking.type=enabled --request-unset temperature --request-unset top_p` |
| MiMo thinking with `max_completion_tokens` | `--request-set extra_body.thinking.type=enabled --request-set max_completion_tokens='${max_output_tokens}' --request-unset max_tokens` |
| OpenRouter reasoning | `--request-set extra_body.reasoning.enabled=true` |
| OpenAI reasoning model on chat completions | `--request-set max_completion_tokens='${max_output_tokens}' --request-unset max_tokens --request-unset temperature --request-unset top_p --request-set reasoning_effort=high` |

## Math Judge

Math scoring always uses an LLM judge. Regex extraction only isolates the
`<answer>` block before judging; there is no regex-only scoring fallback.

Configure the judge in YAML:

```yaml
evaluator_type: math
judge:
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  # api_key: from JUDGE_API_KEY or OPENAI_API_KEY
```

Or on the CLI:

```bash
python main.py \
  --config configs/openai.yaml \
  --api-key "$OPENAI_API_KEY" \
  --judge-model gpt-4o-mini \
  --judge-base-url https://api.openai.com/v1 \
  --judge-api-key "$JUDGE_API_KEY"
```

The judge has its own `request_overrides`, which is useful when using a local
thinking model as the judge and disabling thinking for verdict generation. See
`configs/local.yaml` for a complete example.

Judge identity is recorded in result provenance; judge credentials are never
written to disk.

## Resuming Runs

Pass `--cache` to write completed problems incrementally:

```bash
python main.py --config configs/local.yaml --cache cache/qwen3-4b-math.jsonl
```

If the run is interrupted, rerun the same command with the same cache path.
Completed problems are skipped automatically.

The cache stores an identity header that includes provider, model, dataset,
prompt shape, request overrides, generation settings, output budget, judge
identity, and schema version. If any identity field changes, resume is refused
instead of silently mixing incompatible results.

Cache files are resume state. Use completed JSON files in `results/` for
analysis and reporting.

## Output

Result files are written to `results/` by default and include secret-masked
configuration provenance, per-problem records, and aggregate summary fields:

```json
{
  "summary": {
    "accuracy": 85.5,
    "ock_score": 75.3,
    "total_tokens": 125000,
    "total_prompt_tokens": 8000,
    "total_answer_tokens": 25000,
    "total_reasoning_tokens": 92000,
    "total_output_tokens": 117000,
    "avg_tokens_per_problem": 625.0,
    "error_count": 0
  },
  "results": [
    {
      "problem_id": 1,
      "correct": true,
      "tokens": {
        "prompt_tokens": 80,
        "reasoning_tokens": 420,
        "answer_tokens": 125,
        "output_tokens": 545
      }
    }
  ]
}
```

When comparing runs, keep dataset split, prompt template, provider request
shape, output budget mode, and judge settings consistent.

## Extending OckBench

Providers and evaluators are resolved through registries. Add a module that
registers your implementation, then select it with `--provider` or
`--evaluator-type`.

Custom provider:

```python
from src.core.schemas import ModelResponse, TokenUsage
from src.models.base import BaseModelClient
from src.models.registry import register_provider


@register_provider("my-provider")
class MyClient(BaseModelClient):
    protected_paths = ("model",)
    provider_name = "my-provider"

    def build_request(self, prompt, max_output_tokens):
        return {"model": self.model, "prompt": prompt, "budget": max_output_tokens}

    async def _dispatch(self, request):
        return ModelResponse(
            text="...",
            tokens=TokenUsage(prompt_tokens=0, answer_tokens=0, reasoning_tokens=0),
            latency=0,
            model=self.model,
        )
```

Custom evaluator:

```python
from src.evaluators.base import EvalResult, Evaluator, register_evaluator


@register_evaluator("my-task")
def build(config):
    class MyEvaluator(Evaluator):
        async def evaluate(self, problem, response):
            return EvalResult(
                is_correct=...,
                extracted_answer=...,
                extraction_method="...",
            )

    return MyEvaluator()
```

Validate request construction before running real traffic:

```bash
python main.py --provider my-provider --evaluator-type my-task ... --inspect
```

## Testing

```bash
uv run pytest
```

## Contributing

Issues and pull requests are welcome. Please include the command or YAML config
used for any benchmark behavior you are changing.

## Citation

If you use OckBench in your research, please cite:

```bibtex
@article{du2025ockbench,
  title={OckBench: Measuring the Efficiency of LLM Reasoning},
  author={Du, Zheng and Kang, Hao and Han, Song and Krishna, Tushar and Zhu, Ligeng},
  journal={arXiv preprint arXiv:2511.05722},
  year={2025}
}
```

## License

This repository is available under the MIT license.
