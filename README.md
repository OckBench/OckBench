<!-- markdownlint-disable MD013 MD041 -->
<h1 align="center">OckBench</h1>

<h3 align="center">Efficiency-aware benchmark for LLM reasoning.</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2511.05722"><b>Paper</b></a> |
  <a href="https://ockbench.github.io/"><b>Website</b></a> |
  <a href="https://huggingface.co/ockbench"><b>HuggingFace</b></a>
</p>

OckBench evaluates reasoning models with both correctness and token cost. It
reports raw accuracy plus OckScore:

```text
OckScore = accuracy - 10 * log(avg_tokens_per_problem / 10000 + 1)
```

Built-in tasks:

| Task | Default dataset | Evaluator |
| --- | --- | --- |
| `math` | `data/OckBench_math.jsonl` | LLM judge required |
| `coding` | `data/OckBench_coding.jsonl` | Python test execution |
| `science` | `data/OckBench_science.jsonl` | Multiple-choice answer check |

## Install

OckBench requires Python 3.12 or newer.

```bash
git clone https://github.com/OckBench/OckBench.git
cd OckBench

uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -e .
```

For development:

```bash
uv sync --dev
```

## Run

Every run must choose exactly one budget mode:

- `--max-output-tokens`: fixed output-token budget per problem.
- `--max-context-window`: total context window; OckBench subtracts estimated
  prompt tokens and uses the remainder for output.

### OpenAI-Compatible Endpoint

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Math requires a judge:

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

Coding and science do not need a judge:

```bash
python main.py \
  --model gpt-5.2 \
  --api-key "$OPENAI_API_KEY" \
  --base-url "$OPENAI_BASE_URL" \
  --task science \
  --max-output-tokens 128000
```

### Local vLLM or SGLang

Start an OpenAI-compatible local server, then point OckBench at it:

```bash
vllm serve Qwen/Qwen3-4B --port 8000

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

### YAML Configs

Bundled configs cover OpenAI-compatible chat completions, OpenAI Responses,
Anthropic, Gemini, local servers, and third-party relays:

```bash
python main.py --config configs/openai.yaml --api-key "$OPENAI_API_KEY"
```

Before sending real traffic, inspect the resolved request:

```bash
python main.py --config configs/openai.yaml --api-key "$OPENAI_API_KEY" --inspect
```

CLI flags override YAML fields.

Reasoning and thinking knobs are configurable through `request_overrides`
instead of provider-specific CLI flags. See [Configuration](docs/configuration.md)
for examples such as OpenAI reasoning effort, Gemini thinking budget, and local
Qwen thinking toggles.

## Docs

- [Usage](docs/usage.md): common run commands, local servers, cache/resume.
- [Configuration](docs/configuration.md): task presets, providers, request
  overrides, math judge settings.
- [Results](docs/results.md): output files, token accounting, OckScore, cache
  identity.
- [Extending](docs/extending.md): custom providers and evaluators.
- [Changelog](CHANGELOG.md): notable project changes.

## Test

```bash
uv run pytest
```

## Citation

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
