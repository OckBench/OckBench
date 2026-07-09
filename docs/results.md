# Results

## Output Files

Completed result files are written to `results/` by default. Each file contains:

- `config`: secret-masked run configuration and provenance.
- `results`: one record per problem.
- `summary`: aggregate accuracy, token usage, latency, errors, and OckScore.

Example shape:

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

## Token Fields

- `prompt_tokens`: input tokens.
- `answer_tokens`: visible answer tokens.
- `reasoning_tokens`: hidden or reported reasoning tokens.
- `output_tokens`: `answer_tokens + reasoning_tokens`.
- `total_tokens`: `prompt_tokens + answer_tokens + reasoning_tokens`.

Provider token reporting differs. OckBench normalizes known fields into this
schema so runs can be compared consistently.

## OckScore

OckScore penalizes unnecessary output tokens:

```text
OckScore = accuracy - 10 * log(avg_tokens_per_problem / 10000 + 1)
```

Accuracy is still reported separately. Use both fields when comparing models.

## Cache and Resume

`--cache PATH` writes a JSONL resume file as problems complete:

```bash
python main.py --config configs/local.yaml --cache cache/qwen3-4b-math.jsonl
```

On rerun with the same cache path, completed problems are skipped. The cache
stores an identity header containing provider, model, dataset, prompt shape,
request overrides, generation settings, output budget, judge identity, and
schema version. A configured wall-clock timeout is included because timeout
policy can change retry outcomes. If any identity field changes, resume is
refused instead of mixing incompatible results.

Cache files are not the primary reporting artifact. Use completed JSON files in
`results/` for analysis.

## Comparing Runs

Keep these fields consistent when comparing results:

- Dataset file and split.
- Prompt template and evaluator.
- Provider and request override shape.
- Output budget mode and size.
- Runtime timeout policy, when configured.
- Math judge model, endpoint, and judge request overrides.
