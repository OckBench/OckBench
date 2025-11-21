# OckBench

A powerful LLM benchmarking tool for measuring both **output token count** and **accuracy** on tasks requiring heavy reasoning, such as mathematics, coding, and more.

## Features

- 🚀 **API-Based Benchmarking**: Support for OpenAI, Gemini, and OpenAI-compatible APIs (vLLM, SGLang, etc.)
- 📊 **Token Counting**: Detailed token usage tracking (input, output, reasoning tokens)
- 🎯 **Answer Extraction**: Multi-pattern answer extraction supporting different model output formats
- ⚡ **Concurrent Requests**: Configurable concurrency for efficient benchmarking
- 🔄 **Retry Logic**: Automatic retry with exponential backoff for robustness
- 📝 **Comprehensive Logging**: JSON results with full experiment details and statistics

## Installation

```bash
# Clone the repository
cd /nethome/zdu90/code/OckBench

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using Config Files

```bash
# Set API key
export OPENAI_API_KEY=sk-xxx

# Run with config file
python main.py --config configs/gsm8k_openai.yaml
```

### Using Command Line Arguments

```bash
# OpenAI API
python main.py \
  --dataset data/GSM8K.jsonl \
  --provider openai \
  --model gpt-4 \
  --api-key sk-xxx \
  --temperature 0.0 \
  --concurrency 10

# Gemini API
export GEMINI_API_KEY=your-key
python main.py \
  --dataset data/AIME25.jsonl \
  --provider gemini \
  --model gemini-2.0-flash-exp \
  --concurrency 5

# Local vLLM/SGLang server
python main.py \
  --dataset data/GSM8K.jsonl \
  --provider generic \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1 \
  --concurrency 20
```

## Configuration

### Config File Format

```yaml
# Dataset configuration
dataset_path: data/GSM8K.jsonl
dataset_name: GSM8K

# Model configuration
provider: openai  # openai, gemini, or generic
model: gpt-4
api_key: sk-xxx  # Or use environment variable
# base_url: http://localhost:8000/v1  # For generic provider

# Generation parameters
temperature: 0.0
max_output_tokens: 2048
# OR use max_context_window for dynamic calculation:
# max_context_window: 128000  # Calculates max_output per problem
# reasoning_effort: high  # For o1/o3 models
# top_p: 0.95

# Runtime configuration
concurrency: 10
timeout: 120
max_retries: 3

# Evaluation configuration
evaluator_type: math
```

### Supported Providers

1. **OpenAI**: Official OpenAI API (GPT-4, GPT-3.5, O1, O3, etc.)
2. **Gemini**: Google Gemini API (Gemini Pro, Gemini 1.5 Pro, Gemini 2.0 Flash, etc.)
3. **Generic**: Any OpenAI-compatible API (vLLM, SGLang, local serving)

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `GEMINI_API_KEY`: Google Gemini API key
- `API_KEY`: Generic API key (for custom providers)
- `API_BASE_URL`: Base URL for generic provider

## Dataset Format

Datasets should be in JSONL format with the following fields:

```json
{"problem": "Question text here", "answer": 42, "id": 1}
{"problem": "Another question", "answer": "some answer", "id": 2}
```

Required fields:
- `problem`: The question/problem text (string)
- `answer`: Ground truth answer (any type: number, string, etc.)
- `id`: Problem identifier (any type)

## Answer Extraction

The tool uses multiple patterns to extract answers from model responses:

1. **LaTeX boxed**: `\boxed{answer}`
2. **XML tags**: `<answer>value</answer>`
3. **GSM8K marker**: `#### answer`
4. **Keyword patterns**: "The answer is X", "Final answer: X"
5. **Last number**: Falls back to the last number in the response

This ensures compatibility with different model output formats.

## Results

Results are saved as JSON files in the `results/` directory:

```
results/GSM8K_gpt-4_20241121_143022.json
```

### Result Format

```json
{
  "config": { /* Experiment configuration */ },
  "results": [
    {
      "problem_id": 1,
      "question": "Question text",
      "ground_truth": 42,
      "model_response": "Full model response...",
      "extracted_answer": 42,
      "correct": true,
      "tokens": {
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "reasoning_tokens": 0,
        "total_tokens": 150
      },
      "latency": 2.5,
      "extraction_method": "boxed"
    }
  ],
  "summary": {
    "total_problems": 1000,
    "correct_count": 850,
    "accuracy": 85.0,
    "total_tokens": 150000,
    "total_prompt_tokens": 50000,
    "total_completion_tokens": 100000,
    "total_reasoning_tokens": 0,
    "avg_tokens_per_problem": 150.0,
    "avg_latency": 2.5,
    "total_duration": 250.0,
    "error_count": 0
  },
  "timestamp": "2024-11-21T14:30:22.123456",
  "dataset_name": "GSM8K"
}
```

## Example Configs

The repository includes example configs in `configs/`:

- `gsm8k_openai.yaml`: GSM8K with OpenAI GPT-4
- `aime25.yaml`: AIME25 with OpenAI
- `aime25_gemini.yaml`: AIME25 with Gemini
- `local_vllm.yaml`: Local vLLM/SGLang server
- `o1_reasoning.yaml`: OpenAI O1 with reasoning effort
- `mbpp.yaml`: MBPP coding benchmark

## Advanced Usage

### Dynamic Output Tokens with max_context_window

**NEW FEATURE**: Instead of setting a fixed `max_output_tokens`, you can specify the model's `max_context_window`, and the tool will automatically calculate the maximum output space for each problem based on its input length.

**Benefits:**
- Maximizes output space for each problem
- Adapts to different problem lengths
- Prevents context overflow automatically

**Example config:**
```yaml
model: gpt-4-turbo
max_context_window: 128000  # GPT-4 Turbo's context window
# max_output_tokens is omitted
```

**How it works:**
```
max_output_tokens = max_context_window - input_tokens - safety_buffer(100)
```

**Command line:**
```bash
python main.py \
  --dataset data/AIME25.jsonl \
  --provider openai \
  --model gpt-4-turbo \
  --max-context-window 128000
```

See [MODEL_CONTEXT_WINDOWS.md](MODEL_CONTEXT_WINDOWS.md) for a reference of common model context windows.

### Custom Concurrency and Temperature

```bash
python main.py \
  --config configs/gsm8k_openai.yaml \
  --concurrency 20 \
  --temperature 0.7
```

### O1/O3 Reasoning Models

```bash
python main.py \
  --dataset data/AIME25.jsonl \
  --provider openai \
  --model o1-preview \
  --max-output-tokens 16384 \
  --reasoning-effort high \
  --concurrency 3
```

### Local vLLM Server

First, start your vLLM server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

Then run the benchmark:

```bash
python main.py --config configs/local_vllm.yaml
```

## Token Limits and Context Windows

The tool uses `max_output_tokens` to control the maximum number of tokens in the model's response. If the input exceeds the model's context window, the API will return an error, which will be caught and logged.

You don't need to manually calculate `input + output < context_limit` — the API server handles this automatically.

## Future Work

- [ ] Local batch processing with vLLM/SGLang (non-API mode)
- [ ] HuggingFace dataset integration
- [ ] Code execution evaluator for coding tasks
- [ ] Support for more evaluation metrics
- [ ] Result comparison and analysis tools
- [ ] Web UI for experiment management

## Project Structure

```
OckBench/
├── configs/           # Example configuration files
├── data/              # Dataset files (JSONL format)
├── results/           # Experiment results (JSON)
├── src/
│   ├── core/          # Core logic (runner, config, schemas)
│   ├── loaders/       # Data loaders
│   ├── models/        # API clients
│   ├── evaluators/    # Answer extraction and evaluation
│   └── utils/         # Utilities (logging, etc.)
├── main.py            # CLI entry point
└── requirements.txt   # Python dependencies
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT License
