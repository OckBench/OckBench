# OckBench Local Models

## Setup

```bash
git clone git@github.com:OckBench/OckBench.git && cd OckBench
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# Start model server (I used vLLM, modify as needed)
vllm serve <model_name> --port 8000 --tensor-parallel-size <N>

# Edit BASE_URL in script if needed, then run
./scripts/run_mathbench_local_models.sh
```

## Resume with `--cache`

Each benchmark run supports `--cache <path>` which saves results incrementally to a
JSONL file. If a run is interrupted (e.g. SLURM time limit), re-run the same command
and it will skip already-completed problems automatically.

```bash
# First run (gets interrupted after 150/200 problems)
python main.py --model ... --cache cache/my_model.jsonl

# Re-run the same command — only the remaining 50 problems are processed
python main.py --model ... --cache cache/my_model.jsonl
```

The scripts already use `--cache` by default. Cache files are saved to `cache/`.

**Important for SLURM users:** When your job hits the time limit and you resubmit,
make sure you keep the `--cache` flag pointing to the **same cache file** from the
previous run. The script does this automatically as long as you don't change the model
name or thinking mode — the cache path is deterministic
(`cache/<model_name>_thinking-<true|false>.jsonl`). Just resubmit the same job and it
picks up where it left off.

## Models

Selected by me, feel free to change in the script:

| Model | Max Tokens | Notes |
|-------|------------|-------|
| zai-org/GLM-4.7 | 202752 | thinking=true |
| zai-org/GLM-4.7 | 202752 | thinking=false |
| zai-org/GLM-4.7-Flash | 202752 | thinking=true |
| zai-org/GLM-4.7-Flash | 202752 | thinking=false |
| zai-org/GLM-5 | 202752 | thinking=true |
| zai-org/GLM-5 | 202752 | thinking=false |
| deepseek-ai/DeepSeek-V3.2 | 163840 | thinking=true |
| deepseek-ai/DeepSeek-V3.2 | 163840 | thinking=false |
| moonshotai/Kimi-K2-Instruct | 262144 | |
| moonshotai/Kimi-K2-Thinking | 262144 | |
| moonshotai/Kimi-K2.5 | 262144 | |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | 262144 | |
| Qwen/Qwen3-235B-A22B-Thinking-2507 | 262144 | |

## Output

Results saved to `results/`. Cache files in `cache/`. Send me those JSON files when done.
