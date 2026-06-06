"""Golden / no-semantic-change tests for the scoring core (AC-6).

These pin the published arithmetic so the rewrite cannot silently regress it:
OckScore (incl. zero-token, very-large-token, and empty-result edge cases),
per-provider token normalization (all four providers, incl. the Anthropic
count_tokens-derived answer/reasoning split), and the science / coding scorers
and the math answer-block extractor on fixed fixtures. Everything runs offline.
"""
import asyncio
import math

import pytest

from src.core.schemas import EvaluationResult, TokenUsage
from src.core.scoring import ock_score, summarize
from src.evaluators.code_eval import CodeEvaluator
from src.evaluators.extraction import extract_answer_block
from src.evaluators.science_eval import ScienceEvaluator
from src.utils.usage_normalizer import (
    extract_gemini_usage,
    extract_openai_usage,
    extract_responses_usage,
    normalize_anthropic_usage,
    to_token_usage,
)


# --------------------------------------------------------------------------- #
# OckScore golden values + edge cases
# --------------------------------------------------------------------------- #
def test_ock_score_formula_golden():
    # accuracy - 10 * ln(avg/10000 + 1)
    assert ock_score(85.0, 625.0) == pytest.approx(85.0 - 10 * math.log(625.0 / 10000 + 1))
    assert ock_score(50.0, 45.0) == pytest.approx(50.0 - 10 * math.log(45.0 / 10000 + 1))


def test_ock_score_zero_tokens_is_pure_accuracy():
    # log(0/10000 + 1) == log(1) == 0
    assert ock_score(73.0, 0.0) == 73.0


def test_ock_score_very_large_tokens():
    assert ock_score(100.0, 1_000_000.0) == pytest.approx(100.0 - 10 * math.log(100.0 + 1))


def test_summarize_empty_results():
    summary = summarize([], duration=1.0)
    assert summary.total_problems == 0
    assert summary.accuracy == 0
    assert summary.total_tokens == 0
    assert summary.avg_tokens_per_problem == 0
    assert summary.ock_score == 0.0  # accuracy 0, penalty 0


def _eval_result(*, correct, prompt, answer, reasoning, output, total, latency=0.0):
    return EvaluationResult(
        problem_id="p", question="q", ground_truth="gt", model_response="r",
        correct=correct,
        tokens=TokenUsage(prompt_tokens=prompt, answer_tokens=answer,
                          reasoning_tokens=reasoning, output_tokens=output, total_tokens=total),
        latency=latency,
    )


def test_summarize_golden_rollup():
    results = [
        _eval_result(correct=True, prompt=10, answer=30, reasoning=20, output=50, total=60, latency=1.0),
        _eval_result(correct=False, prompt=5, answer=15, reasoning=5, output=20, total=30, latency=2.0),
    ]
    s = summarize(results, duration=12.5)
    assert (s.total_problems, s.correct_count, s.accuracy) == (2, 1, 50.0)
    assert (s.total_prompt_tokens, s.total_answer_tokens, s.total_reasoning_tokens) == (15, 45, 25)
    assert (s.total_output_tokens, s.total_tokens) == (70, 90)
    assert s.avg_tokens_per_problem == 45.0
    assert s.ock_score == pytest.approx(50.0 - 10 * math.log(45.0 / 10000 + 1))
    assert s.error_count == 0


# --------------------------------------------------------------------------- #
# Per-provider token-normalization golden values (fixed payloads)
# --------------------------------------------------------------------------- #
class _Details:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _OpenAIUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens, reasoning_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        if reasoning_tokens is not None:
            self.completion_tokens_details = _Details(reasoning_tokens)


class _GeminiMeta:
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def test_openai_usage_golden():
    tu = to_token_usage(extract_openai_usage(_OpenAIUsage(5, 100, 105, reasoning_tokens=30)))
    assert (tu.prompt_tokens, tu.reasoning_tokens, tu.answer_tokens, tu.output_tokens, tu.total_tokens) \
        == (5, 30, 70, 100, 105)


def test_responses_usage_golden():
    tu = to_token_usage(extract_responses_usage({
        "input_tokens": 10, "output_tokens": 80, "total_tokens": 90,
        "output_tokens_details": {"reasoning_tokens": 25},
    }))
    assert (tu.prompt_tokens, tu.reasoning_tokens, tu.answer_tokens, tu.output_tokens, tu.total_tokens) \
        == (10, 25, 55, 80, 90)


def test_gemini_usage_golden():
    tu = to_token_usage(extract_gemini_usage(_GeminiMeta(
        prompt_token_count=10, candidates_token_count=40, total_token_count=65, thoughts_token_count=15)))
    assert (tu.prompt_tokens, tu.reasoning_tokens, tu.answer_tokens, tu.output_tokens, tu.total_tokens) \
        == (10, 15, 40, 55, 65)


def test_anthropic_split_golden():
    # answer = min(count_tokens(text), output); reasoning = output - answer.
    async def counter(_text):
        return 30
    tu = to_token_usage(asyncio.run(normalize_anthropic_usage(
        prompt_tokens=5, output_tokens=50, final_text="some answer", count_tokens=counter)))
    assert (tu.prompt_tokens, tu.reasoning_tokens, tu.answer_tokens, tu.output_tokens, tu.total_tokens) \
        == (5, 20, 30, 50, 55)


def test_anthropic_split_clamps_and_empty():
    async def over(_text):
        return 999
    tu = to_token_usage(asyncio.run(normalize_anthropic_usage(
        prompt_tokens=5, output_tokens=50, final_text="x", count_tokens=over)))
    assert (tu.answer_tokens, tu.reasoning_tokens) == (50, 0)  # clamped

    async def never(_text):  # pragma: no cover - must not be awaited
        raise AssertionError("counter must not run for empty text")
    tu2 = to_token_usage(asyncio.run(normalize_anthropic_usage(
        prompt_tokens=5, output_tokens=40, final_text="", count_tokens=never)))
    assert (tu2.answer_tokens, tu2.reasoning_tokens) == (0, 40)


# --------------------------------------------------------------------------- #
# Science multiple-choice scoring (unchanged behavior) on fixed fixtures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("response, gt, expected", [
    ("Reasoning... <answer>C</answer>", "C", True),
    ("The answer is B.", "B", True),
    ("<answer>A</answer>", "B", False),
    ("Therefore the answer is D", "D", True),
])
def test_science_scoring_golden(response, gt, expected):
    # Score via the preserved sync core (extraction + multiple-choice compare).
    ev = ScienceEvaluator()
    extracted, _ = ev.extract_answer(response)
    assert ev.compare_answers(extracted, gt) is expected


# --------------------------------------------------------------------------- #
# Coding test-execution scoring (unchanged behavior) on fixed fixtures
# --------------------------------------------------------------------------- #
def test_coding_scoring_golden_pass():
    ev = CodeEvaluator(timeout=10)
    code, _ = ev.extract_code("<solution>\ndef add(a, b):\n    return a + b\n</solution>")
    all_passed, passed, total, _ = ev.execute_code(code, ["assert add(2, 3) == 5", "assert add(0, 0) == 0"])
    assert all_passed is True and passed == 2 and total == 2


def test_coding_scoring_golden_fail():
    ev = CodeEvaluator(timeout=10)
    code, _ = ev.extract_code("<solution>\ndef add(a, b):\n    return a - b\n</solution>")
    all_passed, passed, total, _ = ev.execute_code(code, ["assert add(2, 3) == 5"])
    assert all_passed is False and passed == 0


# --------------------------------------------------------------------------- #
# Math answer-block extraction (regex demoted to block extraction) golden
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, expected", [
    ("Long reasoning...\n<answer>42</answer>", "42"),
    ("<answer>  x = 3/4  </answer>", "x = 3/4"),
    ("<answer>first</answer> then <answer>final</answer>", "final"),  # last block wins
    ("<ANSWER>7</ANSWER>", "7"),  # case-insensitive
])
def test_math_answer_block_extraction_golden(text, expected):
    content, method = extract_answer_block(text)
    assert content == expected
    assert method == "answer_block"


@pytest.mark.parametrize("text, method", [
    ("", "empty_response"),
    ("   ", "empty_response"),
    ("no tags here, just 42", "no_answer_block"),
    ("<answer></answer>", "no_answer_block"),  # empty block -> not a usable answer
])
def test_math_answer_block_extraction_misses(text, method):
    content, m = extract_answer_block(text)
    assert content is None
    assert m == method
