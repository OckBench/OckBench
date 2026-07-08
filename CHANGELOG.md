# Changelog

## Unreleased — provider & evaluation rewrite (schema_version 2.0)

A bold rewrite of the provider and evaluation layers. The scoring core (OckScore,
token-accounting arithmetic, the prompt/reasoning/answer/output split) is
preserved behaviorally and fenced by golden tests.

### Added
- **Provider registry** (`src/models/registry.py`): every provider is resolved
  through one registry over a common `BaseModelClient` interface. Register a
  custom provider with `@register_provider("name")` — no runner/config edits.
- **Uniform request-shaping** for *all* providers: `request_overrides`
  (`set`/`unset`, `${max_output_tokens}`, per-provider protected paths) now
  applies to `chat_completion`, `openai-responses`, `anthropic`, and `gemini`.
- **Single retry owner**: `BaseModelClient.generate` owns all retry/backoff; every
  provider SDK's auto-retry is disabled (no stacking).
- **Evaluator registry** (`src/evaluators/base.py`) with `@register_evaluator`.
- **Math LLM judge** is the default, required math scorer (regex demoted to
  `<answer>`-block extraction); configure via `judge:` / `--judge-*`.
- **Provenance**: results carry `schema_version`, resolved redacted config (incl.
  judge identity), provider/model + dataset identity + split, token breakdown.
- **Identity-guarded cache**: a cache header records the run-identity tuple;
  resuming with a different identity is refused; the results file is a pure
  aggregation of the cache.
- **`--inspect`** dry-run: prints the sanitized resolved request for any provider
  with no network call.

### Changed / Removed (breaking)
- Removed the dedicated `reasoning_effort` config field and the
  `--reasoning-effort` / `--enable-thinking` flags; express reasoning/thinking via
  `request_overrides` for every provider.
- The client contract is now `build_request` + `_dispatch` (the base owns retry
  and `shape_request`), replacing the old `_call_api(...)` signature.
- Result/cache schemas changed; old result/cache files are not compatible (start
  a fresh `--cache` path). The old post-hoc LLM evaluation script has been
  removed; math scoring now uses the inline judge only.
