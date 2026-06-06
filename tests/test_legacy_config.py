"""Removed legacy reasoning/thinking config keys are rejected, not silently dropped."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.core.config import load_config
from src.core.schemas import (
    BenchmarkConfig,
    ExperimentResult,
    ExperimentSummary,
)


def _base(**extra):
    base = dict(dataset_path="d.jsonl", provider="anthropic", model="claude",
                base_url="https://x", max_output_tokens=100, evaluator_type="science")
    base.update(extra)
    return base


@pytest.mark.parametrize("field, value", [
    ("reasoning_effort", "high"),
    ("enable_thinking", True),
])
def test_legacy_field_rejected_with_migration_hint(field, value):
    with pytest.raises(ValidationError) as exc:
        BenchmarkConfig(**_base(**{field: value}))
    msg = str(exc.value)
    assert field in msg and "request_overrides" in msg


def test_null_legacy_field_is_ignored():
    # A null leftover (from an older serialized config) is not an intentional
    # setting and must not be rejected.
    cfg = BenchmarkConfig(**_base(reasoning_effort=None, enable_thinking=None))
    assert cfg.model == "claude"
    assert not hasattr(cfg, "reasoning_effort")  # field is removed from the schema


def test_legacy_field_rejected_through_load_config():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "c.yaml"
        cfg_path.write_text(
            "provider: openai-responses\ndataset_path: d.jsonl\nmodel: o4-mini\n"
            "base_url: https://x/v1\nmax_output_tokens: 100\nreasoning_effort: high\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError) as exc:
            load_config(str(cfg_path))
    assert "reasoning_effort" in str(exc.value)


def test_historical_result_file_with_legacy_fields_still_loads():
    # Result files written by older versions persist the removed config fields as
    # provenance; loading must tolerate (strip) them.
    cfg = BenchmarkConfig(**_base())
    summary = ExperimentSummary(
        total_problems=0, correct_count=0, accuracy=0, total_tokens=0,
        total_prompt_tokens=0, total_answer_tokens=0, total_reasoning_tokens=0,
        total_output_tokens=0, avg_tokens_per_problem=0, avg_latency=0, total_duration=0,
    )
    result = ExperimentResult(config=cfg, results=[], summary=summary, dataset_name="d")
    data = result.model_dump()
    data["config"]["reasoning_effort"] = "high"   # legacy provenance
    data["config"]["enable_thinking"] = True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old_result.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        loaded = ExperimentResult.load_from_file(str(path))
    assert loaded.config.model == "claude"


def test_bundled_configs_have_no_top_level_legacy_fields():
    # The bundled configs use reasoning_effort only INSIDE request_overrides.set,
    # never at the top level, so they remain valid.
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    for config_file in configs_dir.glob("*.yaml"):
        data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        assert "reasoning_effort" not in data, config_file.name
        assert "enable_thinking" not in data, config_file.name
