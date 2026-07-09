# Usage

## Install

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
uv run pytest
```

## Required Budget Mode

Each run must set exactly one of:

- `--max-output-tokens N`: use a fixed output-token budget per problem.
- `--max-context-window N`: treat `N` as input plus output context; OckBench
  estimates prompt tokens and uses the remaining budget for output.

## OpenAI-Compatible Chat Completions

This path works for OpenAI, local OpenAI-compatible servers, and compatible
relays when you set `--base-url`.

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Math requires a configured LLM judge:

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

Coding and science use deterministic evaluators:

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

## Local vLLM or SGLang

Start the model server first:

```bash
vllm serve Qwen/Qwen3-4B --port 8000
```

Then run OckBench against the OpenAI-compatible endpoint:

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

For deterministic tasks:

```bash
python main.py \
  --model Qwen/Qwen3-4B \
  --api-key dummy \
  --base-url http://localhost:8000/v1 \
  --task science \
  --max-context-window 40960
```

## YAML Configs

Use a bundled config when you want repeatable provider settings:

```bash
python main.py --config configs/openai.yaml --api-key "$OPENAI_API_KEY"
```

Available configs:

| Config | Provider | Intended use |
| --- | --- | --- |
| `configs/openai.yaml` | `chat_completion` | OpenAI-compatible chat completions |
| `configs/openai_responses.yaml` | `openai-responses` | OpenAI `/v1/responses` endpoint |
| `configs/anthropic.yaml` | `anthropic` | Anthropic Messages API |
| `configs/gemini.yaml` | `gemini` | Gemini API |
| `configs/local.yaml` | `chat_completion` | Local vLLM/SGLang server |
| `configs/relay.yaml` | `chat_completion` | Third-party OpenAI-compatible relay |

CLI flags override YAML fields. The effective order is task preset, YAML config,
then CLI overrides.

## Inspect Before Running

`--inspect` resolves config and prints the sanitized outgoing request without
making a network call:

```bash
python main.py --config configs/openai.yaml --api-key "$OPENAI_API_KEY" --inspect
```

API keys, bearer tokens, secret headers, credentials in URLs, and judge
credentials are masked.

## Runtime Timeouts

`--timeout` controls provider/socket timeouts. For streaming providers, it is a
read timeout between chunks.

Use `--wall-clock-timeout N` when you want a hard wall-clock deadline for each
request attempt, for example on low-reasoning runs where a slow trickle should be
retried instead of consuming the full output budget. The wall-clock timeout is
optional and participates in cache identity when set.

## Resume Long Runs

Use `--cache` to write completed problems incrementally:

```bash
python main.py --config configs/local.yaml --cache cache/qwen3-4b-math.jsonl
```

If a run stops, rerun the same command with the same cache path. Completed
problems are skipped automatically. Cache files are resume state; use completed
JSON files in `results/` for analysis and reporting.
