# Quick Start Guide

This guide will help you get started with OckBench in 5 minutes.

## Installation

```bash
cd /nethome/zdu90/code/OckBench
pip install -r requirements.txt
```

## Basic Usage

### 1. Set Your API Key

```bash
# For OpenAI
export OPENAI_API_KEY=sk-your-key-here

# For Gemini
export GEMINI_API_KEY=your-key-here
```

### 2. Run Your First Benchmark

**Option A: Use a config file (recommended)**

```bash
python main.py --config configs/gsm8k_openai.yaml
```

**Option B: Use command line arguments**

```bash
python main.py \
  --dataset data/GSM8K.jsonl \
  --provider openai \
  --model gpt-4 \
  --temperature 0.0 \
  --concurrency 10
```

### 3. View Results

Results are saved in `results/` directory as JSON files:

```bash
ls -lh results/
cat results/GSM8K_gpt-4_*.json | head -50
```

## Example Commands

### Test on AIME25 with GPT-4

```bash
python main.py --config configs/aime25.yaml
```

### Test with Gemini

```bash
export GEMINI_API_KEY=your-key
python main.py --config configs/aime25_gemini.yaml
```

### Test with Local vLLM Server

First, start your vLLM server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Then run the benchmark:

```bash
python main.py --config configs/local_vllm.yaml
```

### Quick Test (First 10 Problems)

To test quickly, you can create a small test file:

```bash
head -10 data/GSM8K.jsonl > data/GSM8K_small.jsonl

python main.py \
  --dataset data/GSM8K_small.jsonl \
  --provider openai \
  --model gpt-3.5-turbo \
  --concurrency 5
```

## Understanding Results

The output will show:

```
Accuracy: 85.00% (850/1000)
Total Tokens: 150,000
  Prompt: 50,000
  Completion: 100,000
  Reasoning: 0
Avg Tokens/Problem: 150.0
Duration: 250.00s
```

The JSON result file contains:
- **config**: Your experiment settings
- **results**: Per-problem results with extracted answers
- **summary**: Aggregate statistics

## Common Issues

### API Key Not Found

```
Error: No API key provided
```

**Solution**: Set the environment variable or add to config:

```bash
export OPENAI_API_KEY=sk-your-key
```

### Rate Limit Errors

```
Error: Rate limit exceeded
```

**Solution**: Reduce concurrency:

```bash
python main.py --config configs/gsm8k_openai.yaml --concurrency 3
```

### Connection Timeout

```
Error: Request timeout
```

**Solution**: Increase timeout:

```bash
python main.py --config configs/aime25.yaml --timeout 300
```

## Pro Tip: Dynamic Output Tokens

Instead of setting fixed `max_output_tokens`, use `max_context_window` to automatically maximize output space:

```yaml
model: gpt-4-turbo
max_context_window: 128000  # Auto-calculates max_output per problem
```

See `configs/aime25_max_context.yaml` for an example and `MODEL_CONTEXT_WINDOWS.md` for model reference.

## Next Steps

1. **Customize configs**: Edit files in `configs/` for your needs
2. **Analyze results**: Use the JSON output for detailed analysis
3. **Compare models**: Run multiple experiments and compare
4. **Add datasets**: Format your own datasets as JSONL

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Review example configs in `configs/`
- Look at [example_usage.py](example_usage.py) for programmatic usage

Happy benchmarking! 🚀

