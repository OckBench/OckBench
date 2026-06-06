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

# Math is scored by an LLM judge (required) — pass the judge model + endpoint.
# Its key resolves from JUDGE_API_KEY or OPENAI_API_KEY (exported above).
python main.py --model gpt-5.2 --api-key $OPENAI_API_KEY --base-url $OPENAI_BASE_URL \
    --task math --max-output-tokens 128000 \
    --judge-model gpt-4o-mini --judge-base-url $OPENAI_BASE_URL

# Coding and science use deterministic scorers (no judge needed).
python main.py --model gpt-5.2 --api-key $OPENAI_API_KEY --base-url $OPENAI_BASE_URL \
    --task coding --max-output-tokens 128000
python main.py --model gpt-5.2 --api-key $OPENAI_API_KEY --base-url $OPENAI_BASE_URL \
    --task science --max-output-tokens 128000
```

Run on a local model via [vLLM](https://docs.vllm.ai/en/latest/usage/) (install vLLM first following their official docs):

```bash
# Start your model server first
vllm serve Qwen/Qwen3-4B --port 8000

# Then run the benchmark (math uses the judge; here the same local server judges,
# with judge thinking disabled — see configs/local.yaml for the YAML form).
python main.py \
    --model Qwen/Qwen3-4B \
    --api-key dummy \
    --base-url http://localhost:8000/v1 \
    --max-context-window 40960 \
    --task math \
    --judge-model Qwen/Qwen3-4B --judge-base-url http://localhost:8000/v1 --judge-api-key dummy

# Coding/science need no judge:
python main.py --model Qwen/Qwen3-4B --api-key dummy --base-url http://localhost:8000/v1 \
    --max-context-window 40960 --task science
```

Use a YAML config for reproducibility (the bundled configs include the judge for
math; `chat_completion` configs still need the model key via `--api-key`):

```bash
python main.py --config configs/openai.yaml --api-key $OPENAI_API_KEY
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
`--temperature`, `--top-p`, `--max-output-tokens`, `--max-context-window`. Each
provider also protects the few request fields it depends on for streaming and
token accounting (e.g. `model`, `messages`, `stream`, `stream_options` for
`chat_completion`); an override targeting a protected field is rejected up front,
naming the field and the provider.

Equivalent YAML (CLI flags take precedence over YAML):

```yaml
request_overrides:
  set:
    extra_body.chat_template_kwargs.enable_thinking: true
  unset:
    - temperature
```

### One request-shaping seam for every provider

Request shaping is uniform across **all** providers — `chat_completion`,
`openai-responses`, `anthropic`, and `gemini`. No provider hard-codes
reasoning/thinking placement; you express it with the same `request_overrides`
mechanism, targeting the field each provider expects:

```yaml
# openai-responses: reasoning effort, drop temperature
provider: openai-responses
request_overrides:
  set: { reasoning.effort: high }
  unset: [temperature]
```

```yaml
# anthropic: effort hint (the default thinking block is overridable too)
provider: anthropic
request_overrides:
  set: { output_config.effort: high }
```

```yaml
# gemini: thinking budget lives under the SDK config dict
provider: gemini
request_overrides:
  set: { config.thinking_config.thinking_budget: 8192 }
```

See `configs/` for a complete, runnable example per provider
(`openai.yaml`, `openai_responses.yaml`, `anthropic.yaml`, `gemini.yaml`,
`local.yaml`, `relay.yaml`).

### Inspecting the resolved request (dry run)

`--inspect` resolves a config and prints the exact outgoing request for any
provider **without making a network call**. Every secret (API key, bearer
tokens, credentials embedded in a `base_url`, secret headers, and the judge's
credentials) is masked, so the output is safe to share:

```bash
python main.py --config configs/openai.yaml --api-key $OPENAI_API_KEY --inspect
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

The request-override mechanism applies uniformly to **every** provider (see "One
request-shaping seam for every provider" above). The dedicated `--reasoning-effort`
and `--enable-thinking` flags — and the per-provider `reasoning_effort` config
field — have been removed; reasoning/thinking is now expressed through
`request_overrides` for all providers.

## Resuming Interrupted Runs

Pass `--cache <path>` to save results incrementally. If a run is interrupted, re-running the same command skips already-completed problems automatically.

```bash
# Start a run (configs/local.yaml already supplies the math judge)
python main.py --config configs/local.yaml --cache cache/qwen3-4b-math.jsonl

# Interrupted? Just re-run the same command — it picks up where it left off
python main.py --config configs/local.yaml --cache cache/qwen3-4b-math.jsonl
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

- Math separates extraction from scoring: a regex isolates the `<answer>` block and an **LLM judge scores it** (the judge is required; see below).
- Coding executes generated code against test cases.
- Science extracts and checks the final multiple-choice answer.

When comparing runs, keep dataset splits and model settings consistent. For example, do not compare a full-dataset run against a Selected-subset run, and keep thinking-enabled runs separate from non-thinking runs.

## Math LLM Judge (required for math)

Math scoring is performed by an LLM judge — regex is used only to extract the
`<answer>` block, which is then handed to the judge. There is **no regex-only
math fallback**: a math run without a configured judge fails fast. Configure the
judge in YAML or on the CLI:

```yaml
evaluator_type: math
judge:
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  # api_key: via env JUDGE_API_KEY or OPENAI_API_KEY
```

```bash
python main.py --config configs/openai.yaml --api-key $OPENAI_API_KEY \
    --judge-model gpt-4o-mini --judge-base-url https://api.openai.com/v1 \
    --judge-api-key $JUDGE_API_KEY
```

The judge accepts its own `request_overrides`, so a local thinking model can be
used as a judge with reasoning disabled (e.g.
`extra_body.chat_template_kwargs.enable_thinking: false`); see `configs/local.yaml`.
The judge's identity (model + endpoint) is recorded in result provenance; its key
is never written to disk. Coding stays code-test evaluated and science stays
multiple-choice evaluated.

> The legacy post-hoc `scripts/llm_eval.py` re-scorer is retained for
> re-evaluating older result files, but the inline judge above is now the
> default, required math scorer.

## Extending OckBench

Providers and evaluators are resolved through registries, so you can add your
own without editing the runner or config schema — just import a module that
registers it.

Custom provider:

```python
from src.models.base import BaseModelClient
from src.models.registry import register_provider
from src.core.schemas import ModelResponse, TokenUsage

@register_provider("my-provider")
class MyClient(BaseModelClient):
    protected_paths = ("model",)        # fields you depend on
    provider_name = "my-provider"

    def build_request(self, prompt, max_output_tokens):
        return {"model": self.model, "prompt": prompt, "budget": max_output_tokens}

    async def _dispatch(self, request):  # the base owns retry; just send + parse
        ...
        return ModelResponse(text=..., tokens=TokenUsage(...), latency=0, model=self.model)
```

Custom task/evaluator:

```python
from src.evaluators.base import Evaluator, EvalResult, register_evaluator

@register_evaluator("my-task")
def build(config):
    class MyEvaluator(Evaluator):
        async def evaluate(self, problem, response):
            return EvalResult(is_correct=..., extracted_answer=..., extraction_method="...")
    return MyEvaluator()
```

Then select it by name: `--provider my-provider` / `--evaluator-type my-task`.
Validate the resolved request first with `--inspect`.

## Reproducibility: provenance & cache identity

Result files carry a `schema_version`, the resolved (secret-masked) config —
including request overrides and the math judge identity — provider/model and
dataset identity, and the normalized token breakdown.

The `--cache` file is the single source of truth: it stores an identity header
(provider, model, dataset+split, prompt/template, resolved request shape,
generation settings, output budget, judge identity, schema version) plus each
problem's full outcome (including the judge verdict). Resuming with a different
identity is refused with a message naming what changed, so results from
different configurations never silently merge; the results file is a pure
aggregation of the cache.

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
