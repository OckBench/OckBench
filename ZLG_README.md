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

## Models

Selected by me, feel free to change in the script:

| Model | Max Tokens | Notes |
|-------|------------|-------|
| zai-org/GLM-4.7 | 202752 | thinking=true |
| zai-org/GLM-4.7 | 202752 | thinking=false |
| zai-org/GLM-4.7-Flash | 202752 | thinking=true |
| zai-org/GLM-4.7-Flash | 202752 | thinking=false |
| deepseek-ai/DeepSeek-V3.2 | 163840 | thinking=true |
| deepseek-ai/DeepSeek-V3.2 | 163840 | thinking=false |
| moonshotai/Kimi-K2-Instruct | 262144 | |
| moonshotai/Kimi-K2-Thinking | 262144 | |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | 262144 | |
| Qwen/Qwen3-235B-A22B-Thinking-2507 | 262144 | |

## Output

Results saved to `results/`. Send me those JSON files when done.
