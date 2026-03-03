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
uv pip install -r requirements.txt
```

You must specify either `--max-output-tokens` or `--max-context-window` to control generation length:

- `--max-output-tokens`: fixed output token budget per problem (use for API models with known limits)
- `--max-context-window`: total context window size; output budget is dynamically calculated as `context_window - prompt_tokens` per problem (use for local models)

Run on OpenAI:

```bash
export OPENAI_API_KEY="sk-..."

python main.py --provider openai --model gpt-5.2 --task math --max-output-tokens 128000
python main.py --provider openai --model gpt-5.2 --task coding --max-output-tokens 128000
python main.py --provider openai --model gpt-5.2 --task science --max-output-tokens 128000
```

Run on a local model via [vLLM](https://docs.vllm.ai/en/latest/usage/) (install vLLM first following their official docs):

```bash
# Start your model server first
vllm serve Qwen/Qwen3-4B --port 8000

# Then run the benchmark
python main.py \
    --provider generic \
    --model Qwen/Qwen3-4B \
    --base-url http://localhost:8000/v1 \
    --max-context-window 40960 \
    --task math
```

Use a YAML config for reproducibility:

```bash
python main.py --config configs/openai.yaml
```

## Resuming Interrupted Runs

Pass `--cache <path>` to save results incrementally. If a run is interrupted, re-running the same command skips already-completed problems automatically.

```bash
# Start a run
python main.py --model Qwen/Qwen3-4B --provider generic \
    --base-url http://localhost:8000/v1 \
    --max-context-window 40960 --task math \
    --cache cache/qwen3-4b-math.jsonl

# Interrupted? Just re-run the same command — it picks up where it left off
python main.py --model Qwen/Qwen3-4B --provider generic \
    --base-url http://localhost:8000/v1 \
    --max-context-window 40960 --task math \
    --cache cache/qwen3-4b-math.jsonl
```

For SLURM users: resubmitting the same job resumes automatically as long as the cache path stays the same.

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

## LLM-based Evaluation

The default evaluators use regex-based answer extraction. For higher accuracy, you can re-evaluate results using an LLM judge with `scripts/llm_eval.py`:

```bash
# Re-evaluate a single result file
python scripts/llm_eval.py results/OckBench_math_gpt-5.2_*.json

# Re-evaluate multiple files with a specific model
python scripts/llm_eval.py results/*.json --model gpt-4o --concurrency 10

# Use a custom output directory
python scripts/llm_eval.py results/*.json --output-dir llm_evaluated/

# Use a local model as judge via OpenAI-compatible endpoint
python scripts/llm_eval.py results/*.json --model Qwen/Qwen3-4B \
    --base-url http://localhost:8000/v1
```

The script produces two output files per input:
- `*_llm_eval.json`: detailed LLM judgments with agreement analysis between regex and LLM evaluators
- `*_llm_rescored.json`: a copy of the original result file with accuracy and OckScore updated using LLM judgments

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
