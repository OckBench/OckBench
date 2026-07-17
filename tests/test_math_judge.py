"""AC-4: extraction separated from scoring; LLM judge is required for math."""
import asyncio
import json

import pytest

from src.core.schemas import BenchmarkConfig, JudgeConfig, Problem
from src.evaluators import get_evaluator
from src.evaluators.judge import JudgeVerdict, LLMJudge, _coerce_correct
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


def test_judge_outage_sets_eval_result_error():
    # A judge that exhausts retries returns a verdict with `error`; the evaluator
    # must surface it as EvalResult.error (not a silent correct=False).
    class _OutageJudge:
        async def score(self, *, question, ground_truth, candidate):
            return JudgeVerdict(correct=False, extracted_answer=None, reasoning="", error="timeout after retries")

    result = asyncio.run(MathEvaluator(_OutageJudge()).evaluate(_problem("6"), "<answer>6</answer>"))
    assert result.error == "timeout after retries"
    assert result.is_correct is False


def test_math_judge_verdict_is_authoritative():
    # Even when the block text equals ground truth, the judge's verdict decides.
    judge = FakeJudge(correct=False)
    ev = MathEvaluator(judge)
    result = asyncio.run(ev.evaluate(_problem("6"), "<answer>6</answer>"))
    assert result.is_correct is False


@pytest.mark.parametrize("response", ["", "   \n\t  "])
def test_empty_response_never_reaches_judge(response):
    # Regression (HLE_Math-240): a budget-exhausted generation with no visible
    # answer went to the judge, whose prompt embeds the ground truth ("F"); the
    # judge hallucinated a match and the blank row was scored correct. An empty
    # response must short-circuit to a scoreable wrong answer without any judge
    # call, regardless of how eager the judge is to say "correct".
    judge = FakeJudge(correct=True)
    result = asyncio.run(MathEvaluator(judge).evaluate(_problem("F"), response))

    assert judge.calls == []  # judge never invoked
    assert result.is_correct is False
    assert result.extracted_answer is None  # ground truth cannot leak in
    assert result.extraction_method == "empty_response"
    # Scoreable outcome, not an infrastructure error: resume must not retry it.
    assert result.error is None


def test_llm_judge_fails_closed_on_empty_candidate():
    # Second line of defense inside the judge itself: even if a caller passes a
    # blank candidate, no LLM call is made and the verdict is incorrect.
    judge = LLMJudge(JudgeConfig(model="m", base_url="https://x/v1", api_key="k"))
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)

        class _Msg:
            content = '{"correct": true, "extracted_answer": "F", "reasoning": "matches"}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
        return _Resp()

    judge.client.chat.completions.create = fake_create
    verdict = asyncio.run(judge.score(question="q", ground_truth="F", candidate="  \n "))

    assert calls == []
    assert verdict.correct is False
    assert verdict.extracted_answer is None
    assert verdict.error is None


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
    # Names all three missing judge fields.
    assert "model" in msg and "base_url" in msg and "api-key" in msg


@pytest.mark.parametrize("judge, missing_term", [
    (JudgeConfig(model="", base_url="https://x/v1", api_key="k"), "judge model"),
    (JudgeConfig(model="m", base_url=None, api_key="k"), "judge endpoint/base_url"),
    (JudgeConfig(model="m", base_url="https://x/v1", api_key=None), "judge api-key"),
])
def test_math_partial_judge_config_fails_fast(judge, missing_term):
    with pytest.raises(ValueError) as exc:
        get_evaluator("math", _math_config(judge=judge))
    assert missing_term in str(exc.value)


def test_math_with_full_judge_config_builds_llm_judge():
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
    assert captured["response_format"] == {"type": "json_object"}
    # The candidate is carried in the judge prompt.
    assert "42" in captured["messages"][0]["content"]


def test_llm_judge_disables_sdk_retry():
    judge = LLMJudge(JudgeConfig(model="m", base_url="https://x/v1", api_key="k"))
    assert judge.client.max_retries == 0


def test_llm_judge_disables_json_mode_when_endpoint_rejects(monkeypatch):
    import src.evaluators.judge as judge_mod

    monkeypatch.setattr(judge_mod.random, "uniform", lambda _lo, _hi: 0)
    judge = LLMJudge(JudgeConfig(model="m", base_url="https://x/v1", api_key="k"), backoff=(0,))
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("unsupported response_format")

        class _Msg:
            content = '{"correct": true, "extracted_answer": "6", "reasoning": "ok"}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
        return _Resp()

    judge.client.chat.completions.create = fake_create
    verdict = asyncio.run(judge.score(question="q", ground_truth="6", candidate="6"))
    assert verdict.correct is True
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False),
    ("true", True), ("True", True), ("yes", True), ("1", True),
    ("false", False), ("False", False), ("no", False), ("0", False), ("", False),
    (1, True), (0, False), (2, False),
    (None, False), ("maybe", False),
])
def test_coerce_correct_strict(value, expected):
    assert _coerce_correct(value) is expected


def test_judge_string_false_is_not_correct():
    # A judge emitting a string boolean "false" must NOT be read as correct.
    judge = LLMJudge(JudgeConfig(model="m", base_url="https://x/v1", api_key="k"))

    async def fake_create(**kwargs):
        class _Msg:
            content = '{"correct": "false", "extracted_answer": "5", "reasoning": "wrong"}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
        return _Resp()

    judge.client.chat.completions.create = fake_create
    verdict = asyncio.run(judge.score(question="q", ground_truth="6", candidate="5"))
    assert verdict.correct is False


def test_parse_json_judgment_recovers_truncated_json():
    # response_format=json_object + verbose reasoning can hit max_tokens, leaving
    # the object unclosed; the verdict fields precede reasoning and must survive.
    from src.evaluators.judge import parse_json_judgment
    truncated = '{"correct": false, "extracted_answer": "x = m^2", "reasoning": "The student\'s answer does not ma'
    v = parse_json_judgment(truncated)
    assert v["correct"] == "false"
    assert v["extracted_answer"] == "x = m^2"
    assert v["reasoning"].endswith("[truncated]")


def test_parse_json_judgment_still_rejects_prose():
    from src.evaluators.judge import parse_json_judgment

    with pytest.raises(json.JSONDecodeError):
        parse_json_judgment("I'm sorry, I can't help with that.")
