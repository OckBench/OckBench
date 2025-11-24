# OckBench

A powerful LLM benchmarking tool for measuring both **efficiency (token count)** and **accuracy** of models on tasks requiring heavy reasoning, such as mathematics, coding, and more. OckBench helps you evaluate how efficiently models solve problems by tracking detailed token usage while simultaneously measuring their accuracy.

## Features

- 🚀 **API-Based Benchmarking**: Support for OpenAI, Gemini, and OpenAI-compatible APIs (vLLM, SGLang, etc.)
- 📊 **Efficiency Measurement**: Detailed token usage tracking (input, output, reasoning tokens) to measure model efficiency
- ✅ **Accuracy Measurement**: Robust evaluation with multiple extraction patterns and code execution for coding tasks
- 🎯 **Output Format Enforcement**: Optional instructions to guide models to format answers consistently (improves extraction accuracy by 40%+)
- 🧠 **Robust Answer Extraction**: 10+ regex patterns for math problems, multi-pattern code extraction for coding problems
- 💻 **Code Evaluation**: Subprocess-based code execution with timeout protection, test validation, and Pass@1 metrics
- ⚡ **Concurrent Execution**: Asynchronous API calls with configurable concurrency and rate limiting
- 🔄 **Automatic Retries**: Exponential backoff retry logic for reliable benchmarking
- 📈 **Dynamic Token Management**: Automatic `max_output_tokens` calculation based on context window size
- 💾 **Comprehensive Logging**: Detailed experiment results with token usage, latency, and error tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/OckBench/OckBench.git
cd OckBench

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using Config Files

```bash
# Set API key
export OPENAI_API_KEY=sk-xxx

# Run with config file
python main.py --config configs/your_config.yaml
```

### Using Command Line Arguments

```bash
# OpenAI API
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider openai \
  --model gpt-4 \
  --api-key sk-xxx \
  --temperature 0.0 \
  --concurrency 10

# Gemini API
export GEMINI_API_KEY=your-key
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider gemini \
  --model gemini-2.0-flash-exp \
  --concurrency 5

# Local vLLM/SGLang server
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider generic \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1 \
  --concurrency 20
```

## Configuration

### Config File Format

```yaml
# Dataset configuration
dataset_path: data/OckBench_math.jsonl
dataset_name: OckBench_math

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
evaluator_type: math  # 'math' or 'code'
enforce_output_format: false  # NEW: Guide models to format answers consistently
# custom_format_instruction: "..."  # Optional custom instruction

# Code evaluation specific (when evaluator_type: code)
execution_timeout: 5  # Timeout for code execution in seconds
include_challenge_tests: true  # Include challenge test cases
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

## Evaluation Types

OckBench supports two types of evaluation:

### Math Problems
- Regex-based answer extraction (10+ patterns)
- Handles various formats: LaTeX boxed, XML tags, natural language
- Direct answer comparison

### Coding Problems (NEW!)
- Multi-pattern code extraction (markdown blocks, raw functions)
- Safe subprocess execution with timeout protection
- Test case validation with Pass@1 metrics
- See [CODING_EXTENSION.md](CODING_EXTENSION.md) for details

## Dataset Format

Datasets should be in JSONL format:

**Math Problems:**
```json
{"problem": "Question text here", "answer": 42, "id": 1}
```

**Coding Problems (OckBench_coding format):**
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

## Answer Extraction

The tool uses multiple patterns to extract answers from model responses:

1. **LaTeX boxed**: `\boxed{answer}`
2. **XML tags**: `<answer>value</answer>`
3. **Marker patterns**: `#### answer`
4. **Keyword patterns**: "The answer is X", "Final answer: X"
5. **Last number**: Falls back to the last number in the response

This ensures compatibility with different model output formats.

## Results

Results are saved as JSON files in the `results/` directory:

```
results/OckBench_math_gpt-4_20241121_143022.json
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

## Example Configs

The repository includes example configs in `configs/`:

- `local_vllm.yaml`: Local vLLM/SGLang server
- `o1_reasoning.yaml`: OpenAI O1 with reasoning effort

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
  --dataset data/OckBench_math.jsonl \
  --provider openai \
  --model gpt-4-turbo \
  --max-context-window 128000
```

See [MODEL_CONTEXT_WINDOWS.md](MODEL_CONTEXT_WINDOWS.md) for a reference of common model context windows.

### Output Format Enforcement (Recommended!)

Add a simple instruction at the start of each problem to guide consistent answer formatting:

```yaml
# In your config file
enforce_output_format: true
```

Or via command line:
```bash
python main.py --config configs/your_config.yaml --enforce-format
```

**How it works**: Prepends a short instruction like "After solving the problem, clearly state your final answer at the end in the format: 'The answer is [NUMBER].'"

**Impact**: Can improve answer extraction accuracy by 40%+ without constraining the model's reasoning process.

**Custom instructions:**
```yaml
enforce_output_format: true
custom_format_instruction: "Please end your response with: Answer = X"
```

### Custom Concurrency and Temperature

```bash
python main.py \
  --config configs/your_config.yaml \
  --concurrency 20 \
  --temperature 0.7
```

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

### Local Model Usage

OckBench supports local models through OpenAI-compatible APIs, such as vLLM, SGLang, or any other local serving solution.

#### Setting Up a Local vLLM Server

1. **Install vLLM** (if not already installed):
```bash
pip install vllm
```

2. **Start the vLLM server**:
```bash
# Basic usage
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# With GPU specification
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --gpu-memory-utilization 0.9

# With tensor parallelism for multi-GPU
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --tensor-parallel-size 2
```

3. **Run the benchmark** using the config file:
```bash
python main.py --config configs/local_vllm.yaml
```

Or via command line:
```bash
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider generic \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --concurrency 20
```

#### Setting Up Other Local Servers

**SGLang:**
```bash
# Start SGLang server
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 8000

# Run benchmark (same as vLLM)
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider generic \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000/v1
```

**Other OpenAI-compatible servers:**
Any server that implements the OpenAI Chat Completions API can be used:
```bash
python main.py \
  --dataset data/OckBench_math.jsonl \
  --provider generic \
  --model your-model-name \
  --base-url http://localhost:8000/v1 \
  --api-key your-api-key-if-needed
```

#### Local Model Configuration Tips

- **Higher concurrency**: Local servers can typically handle much higher concurrency (50-200+) compared to cloud APIs
- **Longer timeouts**: Local models may need more time, especially for complex problems (set `timeout: 3000` or higher)
- **API key**: Many local servers accept any API key or use `dummy` as a placeholder
- **Context window**: Set `max_context_window` to match your model's capabilities for optimal token usage
- **Model name**: Use the exact model identifier that your server expects

Example local model config (`configs/local_vllm.yaml`):
```yaml
provider: generic
model: Qwen/Qwen2.5-7B-Instruct
base_url: http://localhost:8000/v1
api_key: dummy
concurrency: 200  # Higher for local
timeout: 3000     # Longer timeout
max_context_window: 32768  # Match your model
```

## Token Limits and Context Windows

The tool uses `max_output_tokens` to control the maximum number of tokens in the model's response. If the input exceeds the model's context window, the API will return an error, which will be caught and logged.

You don't need to manually calculate `input + output < context_limit` — the API server handles this automatically.

## Future Work

- [ ] Local batch processing with vLLM/SGLang (non-API mode)
- [ ] HuggingFace dataset integration
- [x] Code execution evaluator for coding tasks (OckBench_coding)
- [ ] Support for more coding formats (HumanEval, APPS)
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
