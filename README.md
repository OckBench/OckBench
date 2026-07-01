<!-- markdownlint-disable MD001 MD041 -->
<h1 align="center">OckBench</h1>

<h3 align="center">
Efficiency-aware benchmark for LLM reasoning.
</h3>

<p align="center">
| 📄 <a href="https://arxiv.org/abs/2511.05722"><b>Paper</b></a> | 🌐 <a href="https://ockbench.github.io/"><b>Website</b></a> | 🤗 <a href="https://huggingface.co/ockbench"><b>HuggingFace</b></a> |
</p>

---

## About

OckBench benchmarks LLMs on reasoning tasks while tracking both accuracy and token usage. Most benchmarks only report accuracy — OckBench adds the other axis: how many tokens did the model spend to get there?

We propose **OckScore**, a unified metric that rewards high accuracy achieved with fewer tokens. For more details, please refer to our [paper](https://arxiv.org/abs/2511.05722).

OckBench covers three subfields:

- **Math**: open-ended math problems evaluated by answer extraction
- **Coding**: code generation problems evaluated by executing against test cases
- **Science**: multiple-choice science questions (A/B/C/D)

OckBench is flexible and easy to use:

## Datasets

OckBench includes multiple dataset variants to evaluate LLMs across different problem difficulties and complexities:

- **OckBench_math.jsonl**: Math reasoning problems with clear problem statements
- **OckBench_science.jsonl**: Multiple-choice science questions with consistent formatting
- **OckBench_coding.jsonl**: Standard code generation tasks with well-described specifications
- **OckBench_coding_hard.jsonl**: Challenging coding dataset featuring:
  - Higher problem complexity and ambiguity
  - Mixed-language problems (English, Arabic, Turkish, Chinese)
  - Poorly or incompletely described problem statements
  - Edge cases and unconventional requirements
  
The coding_hard dataset is designed to stress-test LLM reasoning capabilities on more realistic, real-world scenarios where problem statements may be ambiguous or lack complete specifications.

## Features

- Works with OpenAI, Gemini, and any OpenAI-compatible endpoint (vLLM, SGLang, LMDeploy)
- Reasoning tokens tracked separately — distinguish thinking from answering for models like o1/o3
- Fault-tolerant: incremental caching lets interrupted runs resume exactly where they stopped
- YAML configs for reproducible experiments; all parameters also available as CLI flags

## Getting Started

Install from source:

```bash
git clone https://github.com/OckBench/OckBench.git
cd OckBench
uv venv --python 3.12 --managed-python
source .venv/bin/activate
uv pip install -e .
```

You must specify either `--max-output-tokens` or `--max-context-window` to control generation length:

- `--max-output-tokens`: fixed output token budget per problem (use for API models with known limits)
- `--max-context-window`: total context window size; output budget is dynamically calculated as `context_window - prompt_tokens` per problem (use for local models)

Run on OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

python main.py --model gpt-5.2 --api-key $OPENAI_API_KEY --base-url $OPENAI_BASE_URL \
    --task math --max-output-tokens 128000
python main.py --model gpt-5.2 --api-key $OPENAI_API_KEY --base-url $OPENAI_BASE_URL \
    --task coding --max-output-tokens 128000
python main.py --model gpt-5.2 --api-key $OPENAI_API_KEY --base-url $OPENAI_BASE_URL \
    --task science --max-output-tokens 128000
```

Run on a local model via [vLLM](https://docs.vllm.ai/en/latest/usage/) (install vLLM first following their official docs):

```bash
# Start your model server first
vllm serve Qwen/Qwen3-4B --port 8000

# Then run the benchmark
python main.py \
    --model Qwen/Qwen3-4B \
    --api-key dummy \
    --base-url http://localhost:8000/v1 \
    --max-context-window 40960 \
    --task math
```

Use a YAML config for reproducibility:

```bash
python main.py --config configs/openai.yaml
```

## Customizing the Request

Different providers expect the same logical setting (thinking toggles, reasoning
effort, the output-token field, which sampling params are allowed) in different
places in the chat-completions request. Instead of hard-coding per-provider
rules, OckBench lets you shape the outgoing request yourself with three flags:

- `--request-set PATH=JSON_VALUE` — set a field at a dotted `PATH`. The value is
  parsed as JSON (so `true`, `0.9`, `null`, `[...]`, `{...}` keep their types),
  falling back to a plain string. Repeatable.
- `--request-unset PATH` — remove the field at a dotted `PATH`. Repeatable.
- `--request-set-json '<json-object>'` — set many fields at once from a JSON
  object mapping dotted paths to values.

The placeholder `${max_output_tokens}` inside a `--request-set` value is replaced
at request-build time with the per-problem output-token budget (so it still works
with `--max-context-window`).

The standard generation flags still work and land in their usual places:
`--temperature`, `--top-p`, `--max-output-tokens`, `--max-context-window`. Four
fields are managed by OckBench and protected from override (they back streaming
and token accounting): `model`, `messages`, `stream`, `stream_options`.

Equivalent YAML (CLI flags take precedence over YAML):

```yaml
request_overrides:
  set:
    extra_body.chat_template_kwargs.enable_thinking: true
  unset:
    - temperature
```

### Migrating from the old `--enable-thinking` / `--reasoning-effort` flags

These two flags (and the automatic `base_url`/model-name detection that placed
their values) have been removed. Reproduce each former behavior explicitly:

| Former behavior | Override flags |
|-----------------|----------------|
| Local vLLM/SGLang Qwen thinking toggle | `--request-set extra_body.chat_template_kwargs.enable_thinking=true` |
| DeepSeek direct thinking (drops sampling params) | `--request-set extra_body.thinking.type=enabled --request-unset temperature --request-unset top_p` |
| MiMo thinking (uses `max_completion_tokens`) | `--request-set extra_body.thinking.type=enabled --request-set max_completion_tokens='${max_output_tokens}' --request-unset max_tokens` |
| OpenRouter reasoning | `--request-set extra_body.reasoning.enabled=true` (or `--request-set extra_body.reasoning.effort=high`) |
| OpenAI reasoning models (o1/o3/o4/gpt-5): redirect token field, drop sampling params | `--request-set max_completion_tokens='${max_output_tokens}' --request-unset max_tokens --request-unset temperature --request-unset top_p --request-set reasoning_effort=high` |

> Add `--request-unset top_p` (shown above) only when you also pass `--top-p`; the `temperature`/`top_p` drops reproduce providers that ignore sampling params in thinking/reasoning mode. Omit any `--request-unset` for a field you never set.

The request-override mechanism above applies to the `chat_completion` provider.
For the `openai-responses` and `anthropic` providers, reasoning effort is still
set with the `reasoning_effort` config field in YAML (there is no CLI flag), e.g.:

```yaml
provider: anthropic
model: claude-...
reasoning_effort: high   # -> Anthropic output_config.effort / Responses reasoning.effort
```

## Resuming Interrupted Runs

Pass `--cache <path>` to save results incrementally. If a run is interrupted, re-running the same command skips already-completed problems automatically.

```bash
# Start a run
python main.py --model Qwen/Qwen3-4B \
    --api-key dummy --base-url http://localhost:8000/v1 \
    --max-context-window 40960 --task math \
    --cache cache/qwen3-4b-math.jsonl

# Interrupted? Just re-run the same command — it picks up where it left off
python main.py --model Qwen/Qwen3-4B \
    --api-key dummy --base-url http://localhost:8000/v1 \
    --max-context-window 40960 --task math \
    --cache cache/qwen3-4b-math.jsonl
```

For SLURM users: resubmitting the same job resumes automatically as long as the cache path stays the same.

Cache files are resume state only. Use the completed JSON files in `results/` for analysis and reporting.

## Output

Results are saved to `results/` as JSON with per-problem detail and aggregate stats:

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

### Interpreting Results

Each task uses its own evaluator:

- Math uses answer extraction by default. For publication-quality math numbers, run the optional LLM judge described below.
- Coding executes generated code against test cases.
- Science extracts and checks the final multiple-choice answer.

When comparing runs, keep dataset splits and model settings consistent. For example, do not compare a full-dataset run against a Selected-subset run, and keep thinking-enabled runs separate from non-thinking runs.

## LLM-based Evaluation

The default evaluators use regex-based answer extraction. For higher accuracy, you can re-evaluate results using an LLM judge with `scripts/llm_eval.py`:

```bash
# Re-evaluate a single math result file
python scripts/llm_eval.py results/OckBench_math_gpt-5.2_*.json

# Re-evaluate multiple math files with a specific model
python scripts/llm_eval.py results/OckBench_math_*.json --model gpt-4o --concurrency 10

# Use a custom output directory
python scripts/llm_eval.py results/OckBench_math_*.json --output-dir llm_evaluated/

# Use a local model as judge via OpenAI-compatible endpoint
python scripts/llm_eval.py results/OckBench_math_*.json --model Qwen/Qwen3-4B \
    --base-url http://localhost:8000/v1

# For thinking-capable local judges, disable judge thinking if needed
python scripts/llm_eval.py results/OckBench_math_*.json \
    --model Qwen/Qwen3-4B --base-url http://localhost:8000/v1 --disable-thinking
```

The script produces two output files per input:
- `*_llm_eval.json`: detailed LLM judgments with agreement analysis between regex and LLM evaluators
- `*_llm_rescored.json`: a copy of the original result file with accuracy and OckScore updated using LLM judgments

LLM judging is for math accuracy. Coding should remain code-test evaluated, and science should remain multiple-choice evaluated unless a new task-specific judge is added.

## Contributing

We welcome contributions and collaborations. Please open an issue or submit a pull request.

## Citation

If you use OckBench in your research, please cite our paper:

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
