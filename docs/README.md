# OckBench Documentation

Start with the root [README](../README.md) if you only need to install and run
the benchmark. These docs hold the details that make experiments reproducible.

## Pages

- [Usage](usage.md): run commands for API providers, local servers, YAML configs,
  inspection, and cache/resume.
- [Configuration](configuration.md): config precedence, task presets, provider
  keys, output budgets, request overrides, and math judge settings.
- [Results](results.md): result files, token fields, OckScore, and cache
  identity.
- [Extending](extending.md): registering custom providers and evaluators.

## Typical Workflow

1. Pick a task: `math`, `coding`, or `science`.
2. Pick a budget mode: `--max-output-tokens` or `--max-context-window`.
3. Start from a bundled YAML file in [configs](../configs), or pass CLI flags.
4. Run `--inspect` to confirm the final request shape without network traffic.
5. Run with `--cache` for long jobs so interrupted runs can resume.
6. Compare completed JSON files in `results/`, keeping task split, prompt shape,
   provider request shape, budget mode, and judge settings consistent.
