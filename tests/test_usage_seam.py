"""Guard tests for the usage seam wiring and public-contract preservation.

Unlike ``test_usage_normalizer.py`` (which is import-boundary pure), this module
drives a real client over fake streams and inspects the runner, so it imports
provider clients and the runner. It verifies that routing token construction
through the seam left every public surface unchanged: client ``_call_api``
signatures, the ``TokenUsage`` field set, and ``ExperimentSummary`` outputs.
"""
import asyncio
import inspect
import math

from src.core.runner import BenchmarkRunner
from src.core.schemas import BenchmarkConfig, EvaluationResult, TokenUsage
from src.models.anthropic_api import AnthropicClient
from src.models.gemini_api import GeminiClient
from src.models.openai_api import OpenAIClient
from src.models.openai_responses_api import OpenAIResponsesClient

EXPECTED_CALL_API_PARAMS = ["self", "prompt", "temperature", "max_output_tokens", "kwargs"]


# --------------------------------------------------------------------------- #
# Fake streaming primitives for the OpenAI client drive (no network)
# --------------------------------------------------------------------------- #
class _Delta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.model_extra = None


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason


class _Details:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.completion_tokens_details = _Details(reasoning_tokens)


class _Chunk:
    def __init__(self, choices=None, usage=None, model="fake-model"):
        self.choices = choices or []
        self.usage = usage
        self.model = model


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk
        return gen()


# --------------------------------------------------------------------------- #
# AC-6: signature + field-set guards
# --------------------------------------------------------------------------- #
def test_call_api_signatures_unchanged():
    for client_cls in (OpenAIClient, OpenAIResponsesClient, AnthropicClient, GeminiClient):
        params = list(inspect.signature(client_cls._call_api).parameters)
        assert params == EXPECTED_CALL_API_PARAMS, (client_cls.__name__, params)


def test_token_usage_field_set_unchanged():
    assert set(TokenUsage.model_fields) == {
        "prompt_tokens", "answer_tokens", "reasoning_tokens",
        "output_tokens", "total_tokens",
    }


# --------------------------------------------------------------------------- #
# AC-6: client-drive yields identical TokenUsage (routing preserves behavior)
# --------------------------------------------------------------------------- #
def test_openai_client_drive_through_seam():
    client = OpenAIClient(model="m", api_key="k", base_url="https://x/v1")
    chunks = [
        _Chunk(choices=[_Choice(_Delta(reasoning_content="thinking..."))]),
        _Chunk(choices=[_Choice(_Delta(content="Hello "))]),
        _Chunk(choices=[_Choice(_Delta(content="world"), finish_reason="stop")]),
        _Chunk(usage=_Usage(prompt_tokens=10, completion_tokens=50,
                            reasoning_tokens=20, total_tokens=60)),
    ]

    async def fake_create(**kwargs):
        return _FakeStream(chunks)

    client.client.chat.completions.create = fake_create
    response = asyncio.run(client._call_api(prompt="hi", max_output_tokens=100))

    tu = response.tokens
    assert (tu.prompt_tokens, tu.output_tokens, tu.reasoning_tokens,
            tu.answer_tokens, tu.total_tokens) == (10, 50, 20, 30, 60)


def test_openai_extract_tokens_direct_via_seam():
    client = OpenAIClient(model="m", api_key="k", base_url="https://x/v1")
    chunk = _Chunk(usage=_Usage(prompt_tokens=3, completion_tokens=11,
                                reasoning_tokens=4, total_tokens=14))
    tu = client._extract_tokens(chunk)
    assert (tu.prompt_tokens, tu.output_tokens, tu.reasoning_tokens,
            tu.answer_tokens, tu.total_tokens) == (3, 11, 4, 7, 14)


# --------------------------------------------------------------------------- #
# AC-9: ExperimentSummary computed outputs preserved
# --------------------------------------------------------------------------- #
def _eval_result(*, correct, prompt, answer, reasoning, output, total, latency):
    return EvaluationResult(
        problem_id="p",
        question="q",
        ground_truth="gt",
        model_response="resp",
        correct=correct,
        tokens=TokenUsage(prompt_tokens=prompt, answer_tokens=answer,
                          reasoning_tokens=reasoning, output_tokens=output,
                          total_tokens=total),
        latency=latency,
    )


def test_compute_summary_preserved():
    config = BenchmarkConfig(dataset_path="x", provider="gemini", model="m",
                            max_output_tokens=100)
    runner = BenchmarkRunner(config)

    results = [
        _eval_result(correct=True, prompt=10, answer=30, reasoning=20,
                     output=50, total=60, latency=1.0),
        _eval_result(correct=False, prompt=5, answer=15, reasoning=5,
                     output=20, total=30, latency=2.0),
    ]
    summary = runner._compute_summary(results, duration=12.5)

    assert summary.total_problems == 2
    assert summary.correct_count == 1
    assert summary.accuracy == 50.0
    assert summary.total_prompt_tokens == 15
    assert summary.total_answer_tokens == 45
    assert summary.total_reasoning_tokens == 25
    assert summary.total_output_tokens == 70
    assert summary.total_tokens == 90

    expected_avg = 90 / 2
    expected_ock = 50.0 - 10 * math.log(expected_avg / 10000 + 1)
    assert summary.avg_tokens_per_problem == expected_avg
    assert summary.ock_score == expected_ock
    assert summary.error_count == 0
