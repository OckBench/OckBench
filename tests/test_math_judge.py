"""AC-4: extraction separated from scoring; LLM judge is required for math."""
import asyncio

import pytest

from src.core.schemas import BenchmarkConfig, JudgeConfig, Problem
from src.evaluators import get_evaluator
from src.evaluators.judge import JudgeVerdict, LLMJudge
from src.evaluators.math_eval import MathEvaluator


class FakeJudge:
    """Records exactly what the scorer received and returns a fixed verdict."""

    def __init__(self, correct=True):
        self.calls = []
        self.correct = correct

    async def score(self, *, question, ground_truth, candidate):
        self.calls.append({"question": question, "ground_truth": ground_truth, "candidate": candidate})
        return JudgeVerdict(correct=self.correct, extracted_answer=candidate, reasoning="ok")


def _problem(answer="6"):
    return Problem(problem="Find n.", answer=answer, id="p1")


def test_math_scores_via_judge_over_extracted_block():
    judge = FakeJudge(correct=True)
    ev = MathEvaluator(judge)
    response = "lots of reasoning...\n<answer>6</answer>"
    result = asyncio.run(ev.evaluate(_problem("6"), response))

    assert result.is_correct is True
    assert len(judge.calls) == 1  # judge WAS invoked (no silent regex fallback)
    # Scorer received the extracted block, not the entire raw transcript.
    assert judge.calls[0]["candidate"] == "6"
    assert "lots of reasoning" not in judge.calls[0]["candidate"]
    assert result.extraction_method == "answer_block"


def test_math_judge_verdict_is_authoritative():
    # Even when the block text equals ground truth, the judge's verdict decides.
    judge = FakeJudge(correct=False)
    ev = MathEvaluator(judge)
    result = asyncio.run(ev.evaluate(_problem("6"), "<answer>6</answer>"))
    assert result.is_correct is False


def test_math_falls_back_to_full_response_only_when_no_block():
    judge = FakeJudge(correct=True)
    ev = MathEvaluator(judge)
    asyncio.run(ev.evaluate(_problem("6"), "the answer is 6"))  # no <answer> block
    assert judge.calls[0]["candidate"] == "the answer is 6"


def _math_config(judge=None):
    return BenchmarkConfig(
        dataset_path="d.jsonl", provider="gemini", model="m",
        max_output_tokens=100, evaluator_type="math", judge=judge,
    )


def test_math_without_judge_fails_fast():
    with pytest.raises(ValueError) as exc:
        get_evaluator("math", _math_config(judge=None))
    msg = str(exc.value).lower()
    assert "judge" in msg
    assert "model" in msg or "endpoint" in msg or "url" in msg


def test_math_with_judge_config_builds_llm_judge():
    cfg = _math_config(judge=JudgeConfig(model="gpt-4o-mini", base_url="https://x/v1", api_key="k"))
    ev = get_evaluator("math", cfg)
    assert isinstance(ev, MathEvaluator)
    assert isinstance(ev.judge, LLMJudge)


def test_llm_judge_builds_request_and_parses_offline():
    cfg = JudgeConfig(
        model="judge-m", base_url="https://x/v1", api_key="k",
        request_overrides={"set": {"extra_body.chat_template_kwargs.enable_thinking": False}, "unset": []},
    )
    judge = LLMJudge(cfg)
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)

        class _Msg:
            content = '{"correct": true, "extracted_answer": "42", "reasoning": "match"}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
        return _Resp()

    judge.client.chat.completions.create = fake_create
    verdict = asyncio.run(judge.score(question="q", ground_truth="42", candidate="42"))

    assert verdict.correct is True and verdict.extracted_answer == "42"
    assert captured["model"] == "judge-m"
    assert captured["max_tokens"] == 500
    # Judge request_overrides applied (disable a local thinking model).
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    # The candidate is carried in the judge prompt.
    assert "42" in captured["messages"][0]["content"]


def test_llm_judge_disables_sdk_retry():
    judge = LLMJudge(JudgeConfig(model="m", base_url="https://x/v1", api_key="k"))
    assert judge.client.max_retries == 0
