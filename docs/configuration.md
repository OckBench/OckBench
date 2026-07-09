# Configuration

## Precedence

Configuration is resolved in this order:

1. `--task` preset.
2. YAML file from `--config`.
3. CLI flags.

CLI flags override YAML fields. If a CLI budget flag is set, the opposite budget
field from YAML is removed.

## Task Presets

`--task` selects the default dataset and evaluator:

| `--task` | Default dataset | Default evaluator |
| --- | --- | --- |
| `math` | `data/OckBench_math.jsonl` | `math` |
| `coding` | `data/OckBench_coding.jsonl` | `code` |
| `science` | `data/OckBench_science.jsonl` | `science` |

Override the dataset with `--dataset-path`, `--dataset-name`, and
`--dataset-split`. Selected and mini splits are included under [data](../data).

## Providers and Keys

Built-in providers:

| Provider | API surface | Environment fallback |
| --- | --- | --- |
| `chat_completion` | OpenAI-compatible chat completions | none; pass `--api-key` and `--base-url` |
| `openai-responses` | OpenAI Responses API | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `gemini` | Gemini API | `GEMINI_API_KEY` |

The math judge key resolves from `JUDGE_API_KEY`, then `OPENAI_API_KEY`, unless
you pass `--judge-api-key` or set `judge.api_key` in YAML.

## Output Budgets

Exactly one budget mode is required:

- `max_output_tokens`: fixed output-token cap for every problem.
- `max_context_window`: total input plus output context. OckBench estimates
  prompt tokens per problem and uses the remaining budget for output.

In request overrides, `${max_output_tokens}` is replaced at request-build time
with the effective per-problem output budget.

## Runtime Timeouts

OckBench has two request timeout layers:

- `timeout`: provider/socket timeout. Streaming providers use it as a read
  timeout between chunks, so a slow stream that keeps sending bytes can continue.
- `wall_clock_timeout`: optional total wall-clock deadline for one request
  attempt. If an attempt exceeds it, OckBench treats that attempt as retryable
  and continues through `max_retries`.

`wall_clock_timeout` defaults to unset. When set, it is part of cache identity
because timeout-driven retries can change outcomes.

## Request Overrides

Use request overrides to place provider-specific reasoning, thinking, sampling,
and budget fields without adding one-off CLI flags:

- `--request-set PATH=JSON_VALUE`: set a dotted request field.
- `--request-unset PATH`: remove a dotted request field.
- `--request-set-json '{"path": value}'`: set several fields at once.

YAML form:

```yaml
request_overrides:
  set:
    max_completion_tokens: ${max_output_tokens}
    reasoning_effort: high
  unset:
    - max_tokens
    - temperature
```

Common examples:

```bash
# Qwen thinking through vLLM/SGLang
python main.py ... \
  --request-set extra_body.chat_template_kwargs.enable_thinking=true
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

Overrides cannot replace protected fields such as `model`, `messages`, `input`,
`stream`, or stream options. Invalid overrides fail before the run starts.

## Math Judge

Math scoring always uses an LLM judge. Regex extraction only isolates an
`<answer>` block before judging; there is no regex-only scoring fallback.

YAML:

```yaml
evaluator_type: math
judge:
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  # api_key: from JUDGE_API_KEY or OPENAI_API_KEY
```

CLI:

```bash
python main.py \
  --config configs/openai.yaml \
  --api-key "$OPENAI_API_KEY" \
  --judge-model gpt-4o-mini \
  --judge-base-url https://api.openai.com/v1 \
  --judge-api-key "$JUDGE_API_KEY"
```

The judge also supports `request_overrides`, which is useful when a local
thinking model should generate direct verdicts.

## Removed Reasoning Flags

The older `--enable-thinking` and `--reasoning-effort` flags were removed. Use
request overrides instead:

| Former behavior | Current override |
| --- | --- |
| Local Qwen thinking toggle | `--request-set extra_body.chat_template_kwargs.enable_thinking=true` |
| DeepSeek thinking, no sampling params | `--request-set extra_body.thinking.type=enabled --request-unset temperature --request-unset top_p` |
| MiMo thinking with `max_completion_tokens` | `--request-set extra_body.thinking.type=enabled --request-set max_completion_tokens='${max_output_tokens}' --request-unset max_tokens` |
| OpenRouter reasoning | `--request-set extra_body.reasoning.enabled=true` |
| OpenAI reasoning model on chat completions | `--request-set max_completion_tokens='${max_output_tokens}' --request-unset max_tokens --request-unset temperature --request-unset top_p --request-set reasoning_effort=high` |
