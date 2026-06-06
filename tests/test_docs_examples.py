"""AC-10: documented config examples validate via the inspect path (offline)."""
import socket
from pathlib import Path

import pytest
import yaml

import src.evaluators.base as ev_base
import src.models.registry as registry
from src.core.config import load_config
from src.core.inspect import build_inspection, format_inspection
from src.core.schemas import BenchmarkConfig, ModelResponse, TokenUsage
from src.evaluators import EvalResult, Evaluator, get_evaluator
from src.models.base import BaseModelClient

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIG_FILES = sorted(CONFIGS_DIR.glob("*.yaml"))
REMOVED_TOP_LEVEL_FIELDS = ("reasoning_effort", "enable_thinking")


def test_configs_exist():
    names = {p.name for p in CONFIG_FILES}
    # Official providers, local server, and a third-party relay are all documented.
    assert {"openai.yaml", "gemini.yaml", "anthropic.yaml",
            "openai_responses.yaml", "local.yaml", "relay.yaml"} <= names


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=lambda p: p.name)
def test_bundled_config_validates_via_inspect(config_file, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError(f"{config_file.name} opened a socket during inspect")
    monkeypatch.setattr(socket, "socket", _boom)

    # dummy api_key satisfies the chat_completion requirement; harmless elsewhere.
    cfg = load_config(str(config_file), api_key="dummy-key")
    inspection = build_inspection(cfg)
    blob = format_inspection(inspection)
    assert inspection["provider"] == cfg.provider
    assert "dummy-key" not in blob  # the key is masked even in inspect output


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=lambda p: p.name)
def test_no_removed_flags_in_configs(config_file):
    data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    for removed in REMOVED_TOP_LEVEL_FIELDS:
        assert removed not in data, f"{config_file.name} references removed field '{removed}'"


def test_documented_custom_provider_pattern():
    # Mirrors the README "registering a custom provider" example.
    name = "doc-echo-provider"
    try:
        @registry.register_provider(name)
        class _Echo(BaseModelClient):
            protected_paths = ("model",)
            provider_name = name

            def build_request(self, prompt, max_output_tokens):
                return {"model": self.model, "prompt": prompt}

            async def _dispatch(self, request):
                return ModelResponse(text="ok", tokens=TokenUsage(), latency=0, model=self.model)

        cfg = BenchmarkConfig(dataset_path="d.jsonl", provider=name, model="m",
                              max_output_tokens=100, evaluator_type="science")
        inspection = build_inspection(cfg)
        assert inspection["provider"] == name
    finally:
        registry._PROVIDER_REGISTRY.pop(name, None)


def test_documented_custom_evaluator_pattern():
    # Mirrors the README "registering a custom task/evaluator" example.
    name = "doc-exact-eval"
    try:
        @ev_base.register_evaluator(name)
        def _factory(config):
            class _Exact(Evaluator):
                async def evaluate(self, problem, response):
                    return EvalResult(is_correct=str(problem.answer) in response,
                                      extracted_answer=problem.answer, extraction_method="exact")
            return _Exact()

        cfg = BenchmarkConfig(dataset_path="d.jsonl", provider="gemini", model="m",
                              max_output_tokens=100, evaluator_type=name)
        assert get_evaluator(name, cfg) is not None
    finally:
        ev_base._EVALUATOR_REGISTRY.pop(name, None)
