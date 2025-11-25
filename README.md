# OckBench

A powerful LLM benchmarking tool for measuring both **efficiency (token count)** and **accuracy** of models on tasks requiring heavy reasoning, such as mathematics, coding, and more. OckBench helps you evaluate how efficiently models solve problems by tracking detailed token usage while simultaneously measuring their accuracy.

![Efficiency vs Accuracy Tradeoff](teaser.png)

*OckBench visualizes the efficiency-accuracy tradeoff across different models, helping you understand which models achieve the best balance between token usage and problem-solving accuracy.*

## Features

- 📊 **Dual Measurement**: Track both **efficiency (token count)** and **accuracy** in a single benchmark
- 🚀 **Multi-Provider Support**: OpenAI, Gemini, and any OpenAI-compatible API (vLLM, SGLang, local models)
- 🎯 **Format Enforcement**: Optional instructions to guide models for consistent output (improves extraction accuracy by 40%+)
- 🧠 **Robust Extraction**: 10+ regex patterns for math, multi-pattern code extraction for coding tasks
- 💻 **Code Evaluation**: Safe subprocess execution with timeout protection and Pass@1 metrics
- ⚡ **High Performance**: Asynchronous API calls with configurable concurrency
- 📈 **Dynamic Token Management**: Automatic `max_output_tokens` calculation based on context window size
- 💾 **Comprehensive Results**: Detailed JSON outputs with token usage, latency, and error tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/OckBench/OckBench.git
cd OckBench

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Commercial API Example (OpenAI)

```bash
# Set API key
export OPENAI_API_KEY=sk-xxx

# Run with config file
python main.py --config configs/ockbench_math_openai.yaml
```

### Local Model Example

```bash
# 1. Start vLLM server (in another terminal)
vllm serve Qwen/Qwen3-4B --port 8000

# 2. Run benchmark
python main.py --config configs/ockbench_math_local.yaml
```

## Examples

### Commercial APIs

**OpenAI (Math):**
```bash
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider openai \
  --model gpt-5 \
  --api-key sk-xxx \
  --concurrency 10
```

**Gemini (Math):**
```bash
export GEMINI_API_KEY=your-key
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider gemini \
  --model gemini-2.5-flash \
  --concurrency 5
```

**OpenAI (Coding):**
```bash
python main.py --config configs/ockbench_coding_openai.yaml
```

### Local Models

**Using vLLM:**
```bash
# Start server
vllm serve Qwen/Qwen3-4B --port 8000

# Run benchmark
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider generic \
  --model Qwen/Qwen3-4B \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --concurrency 200
```

**Using SGLang:**
```bash
# Start server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --port 8000

# Run benchmark (same as vLLM)
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider generic \
  --model Qwen/Qwen3-4B \
  --base-url http://localhost:8000/v1
```

## Configuration

### Config File Format

OckBench uses YAML configuration files for easy experiment management:

```yaml
# Dataset configuration
dataset_path: data/OckBench_math.jsonl
dataset_name: OckBench_math

# Model configuration
provider: openai  # openai, gemini, or generic
model: gpt-5
# api_key: sk-xxx  # Or use environment variable

# Generation parameters
temperature: 0.0
max_output_tokens: 4096
# OR use max_context_window for dynamic calculation:
# max_context_window: 128000  # Calculates max_output per problem

# Runtime configuration
concurrency: 10
timeout: 120
max_retries: 3

# Evaluation configuration
evaluator_type: math  # 'math' or 'code'
enforce_output_format: true  # Recommended: improves extraction accuracy

# Code evaluation specific (when evaluator_type: code)
execution_timeout: 5  # Timeout for code execution in seconds
include_challenge_tests: true  # Include challenge test cases
```

### Example Configs

The repository includes ready-to-use configs in `configs/`:

- **Commercial APIs:**
  - `ockbench_math_openai.yaml` - OpenAI API for math problems
  - `ockbench_math_gemini.yaml` - Gemini API for math problems
  - `ockbench_coding_openai.yaml` - OpenAI API for coding problems

- **Local Models:**
  - `ockbench_math_local.yaml` - Local vLLM/SGLang for math
  - `ockbench_coding_local.yaml` - Local vLLM/SGLang for coding

### Supported Providers

1. **OpenAI**: GPT-4, GPT-5, O1, O3, and other OpenAI models
2. **Gemini**: Gemini 2.5 Flash, Gemini 2.0, Gemini 1.5 Pro, etc.
3. **Generic**: Any OpenAI-compatible API (vLLM, SGLang, local serving)

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `GEMINI_API_KEY`: Google Gemini API key
- `API_KEY`: Generic API key (for custom providers)

## Dataset Format

### Math Problems

```json
{"problem": "Question text here", "answer": 42, "id": 1}
```

### Coding Problems

```json
{
  "doc_id": 0,
  "doc": {
    "task_id": 11,
    "text": "Write a function...",
    "code": "def solution(): ...",
    "test_list": ["assert func() == expected"],
    "challenge_test_list": ["assert func('edge') == expected"]
  }
}
```

Auto-detects format based on filename (contains 'coding') or structure.

## Results

Results are saved as JSON files in the `results/` directory:

```
results/OckBench_math_gpt-5_20241121_143022.json
```

### Result Summary

After running a benchmark, you'll see a summary like:

```
EXPERIMENT SUMMARY
================================================================================
Dataset: OckBench_math
Model: gpt-5
Accuracy: 85.0% (850/1000)
Total Tokens: 150,000
  Prompt: 50,000
  Answer: 100,000
  Reasoning: 0
  Output: 100,000
Avg Tokens/Problem: 150.0
Duration: 250.0s
================================================================================
```

### Result Format

The JSON output includes detailed information for each problem:

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
        "answer_tokens": 100,
        "reasoning_tokens": 0,
        "output_tokens": 100,
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
    "total_answer_tokens": 100000,
    "total_reasoning_tokens": 0,
    "total_output_tokens": 100000,
    "avg_tokens_per_problem": 150.0,
    "avg_latency": 2.5,
    "total_duration": 250.0,
    "error_count": 0
  },
  "timestamp": "2024-11-21T14:30:22.123456",
  "dataset_name": "OckBench_math"
}
```

## Advanced Usage

### Dynamic Output Tokens with max_context_window

Instead of setting a fixed `max_output_tokens`, specify the model's `max_context_window` for automatic calculation:

```yaml
model: gpt-4-turbo
max_context_window: 128000  # GPT-4 Turbo's context window
# max_output_tokens is automatically calculated per problem
```

**Benefits:**
- Maximizes output space for each problem
- Adapts to different problem lengths
- Prevents context overflow automatically

**How it works:**
```
max_output_tokens = max_context_window - input_tokens - safety_buffer(100)
```

### Output Format Enforcement

Improve answer extraction accuracy by guiding models to format answers consistently:

```yaml
enforce_output_format: true
```

Or via command line:
```bash
python main.py --config configs/ockbench_math_openai.yaml --enforce-format
```

**Impact**: Can improve answer extraction accuracy by 40%+ without constraining the model's reasoning process.

### O1/O3 Reasoning Models

```bash
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider openai \
  --model o1-preview \
  --max-output-tokens 16384 \
  --reasoning-effort high \
  --concurrency 3
```

### Local Model Configuration Tips

- **Higher concurrency**: Local servers can handle much higher concurrency (50-200+) compared to cloud APIs
- **Longer timeouts**: Set `timeout: 3000` or higher for complex problems
- **API key**: Many local servers accept any API key or use `dummy` as a placeholder
- **Context window**: Set `max_context_window` to match your model's capabilities

## Evaluation Types

### Math Problems
- Regex-based answer extraction (10+ patterns)
- Handles various formats: LaTeX boxed, XML tags, natural language
- Direct answer comparison

### Coding Problems
- Multi-pattern code extraction (markdown blocks, raw functions)
- Safe subprocess execution with timeout protection
- Test case validation with Pass@1 metrics

## Answer Extraction

The tool uses multiple patterns to extract answers from model responses:

1. **LaTeX boxed**: `\boxed{answer}`
2. **XML tags**: `<answer>value</answer>`
3. **Marker patterns**: `#### answer`
4. **Keyword patterns**: "The answer is X", "Final answer: X"
5. **Last number**: Falls back to the last number in the response

This ensures compatibility with different model output formats.

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
