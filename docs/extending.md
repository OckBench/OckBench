# Extending OckBench

Providers and evaluators are resolved through registries. Add a module that
registers your implementation, import it before building the config, then select
it with `--provider` or `--evaluator-type`.

Validate request construction with `--inspect` before running real traffic:

```bash
python main.py --provider my-provider --evaluator-type my-task ... --inspect
```

## Custom Provider

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

`protected_paths` prevents user overrides from replacing fields the provider
must own, such as `model`, `messages`, `input`, or streaming controls.

## Custom Evaluator

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

Evaluator names can be selected with `--evaluator-type`. Task presets set the
built-in evaluator automatically, but explicit CLI and YAML values override the
preset.

## Development Checks

```bash
uv run pytest
```

When changing benchmark behavior, include the command or YAML config used to
produce the result.
