"""AC-7: result provenance & schema versioning (offline)."""
import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.schemas import (
    SCHEMA_VERSION,
    BenchmarkConfig,
    EvaluationResult,
    ExperimentResult,
    JudgeConfig,
    TokenUsage,
)
from src.core.scoring import summarize


def _make_experiment():
    cfg = BenchmarkConfig(
        dataset_path="data/OckBench_Math_Selected.jsonl",
        dataset_name="OckBench_math", dataset_split="Selected",
        provider="chat_completion", model="gpt-x",
        base_url="https://api.openai.com/v1", api_key="sk-SECRET",
        max_output_tokens=1000, evaluator_type="math",
        request_overrides={"set": {"reasoning_effort": "medium", "headers.x-api-key": "HDR-SECRET"},
                           "unset": ["temperature"]},
        judge=JudgeConfig(model="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key="judge-SECRET"),
    )
    results = [EvaluationResult(
        problem_id="AMO-0", question="q", ground_truth="6", model_response="<answer>6</answer>",
        extracted_answer="6", correct=True,
        tokens=TokenUsage(prompt_tokens=80, answer_tokens=125, reasoning_tokens=420,
                          output_tokens=545, total_tokens=625),
        latency=1.0, extraction_method="answer_block", judge_reasoning="match",
    )]
    summary = summarize(results, duration=1.0)
    return ExperimentResult(config=cfg, results=results, summary=summary, dataset_name="OckBench_math")


def test_result_file_carries_full_provenance():
    exp = _make_experiment()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.json"
        exp.save_to_file(str(path))
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)

    assert data["schema_version"] == SCHEMA_VERSION
    cfg = data["config"]
    # Provider + model identity
    assert cfg["provider"] == "chat_completion" and cfg["model"] == "gpt-x"
    # Dataset identity + split
    assert cfg["dataset_name"] == "OckBench_math" and cfg["dataset_split"] == "Selected"
    assert cfg["dataset_path"].endswith("OckBench_Math_Selected.jsonl")
    # Resolved request overrides recorded
    assert cfg["request_overrides"]["set"]["reasoning_effort"] == "medium"
    assert "temperature" in cfg["request_overrides"]["unset"]
    # Judge identity present: model + endpoint class, key masked
    assert cfg["judge"]["model"] == "gpt-4o-mini"
    assert cfg["judge"]["base_url"] == "https://api.openai.com/v1"
    assert cfg["judge"]["api_key"] == "***MASKED***"
    # Secrets masked everywhere
    assert cfg["api_key"] == "***MASKED***"
    assert cfg["request_overrides"]["set"]["headers.x-api-key"] == "***MASKED***"
    assert "SECRET" not in raw

    # Normalized token breakdown
    s = data["summary"]
    for key in ("total_prompt_tokens", "total_answer_tokens", "total_reasoning_tokens",
                "total_output_tokens", "total_tokens"):
        assert key in s
    assert data["results"][0]["tokens"]["reasoning_tokens"] == 420


def test_result_missing_identity_fails_to_load():
    exp = _make_experiment()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "result.json"
        exp.save_to_file(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        # Remove provider/model identity -> must fail validation on load.
        data["config"].pop("provider")
        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValidationError):
            ExperimentResult.load_from_file(str(bad))


def test_token_usage_field_set_unchanged():
    # Contract guard: the normalized token split is exactly these five counts,
    # plus the diagnostic markers for estimated splits and provider-unattributed
    # totals (never used in scoring).
    assert set(TokenUsage.model_fields) == {
        "prompt_tokens", "answer_tokens", "reasoning_tokens", "output_tokens", "total_tokens",
        "answer_tokens_estimated", "unattributed_tokens",
    }
