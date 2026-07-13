"""AC-8: cache-resume identity safety + results derive from the cache (offline)."""
import json
import tempfile
from pathlib import Path

import pytest

import src.evaluators.math_eval as math_eval
import src.models.registry as registry
from src.core.cache import CacheIdentityMismatch, RunCache, aggregate_cache_file
from src.core.identity import compute_run_identity
from src.core.runner import BenchmarkRunner
from src.core.schemas import BenchmarkConfig, JudgeConfig, ModelResponse, TokenUsage
from src.evaluators.judge import JudgeVerdict
from src.models.base import BaseModelClient
from tests.transport_fakes import EmptyStrError


# --------------------------------------------------------------------------- #
# Test doubles: a fixed-answer provider and a call-counting judge
# --------------------------------------------------------------------------- #
class _CountingJudge:
    def __init__(self):
        self.calls = 0

    async def score(self, *, question, ground_truth, candidate):
        self.calls += 1
        return JudgeVerdict(correct=(str(candidate) == str(ground_truth)),
                            extracted_answer=str(candidate), reasoning="")


class _ErroringJudge:
    """Simulates a judge outage: always returns a verdict with an error."""

    def __init__(self):
        self.calls = 0

    async def score(self, *, question, ground_truth, candidate):
        self.calls += 1
        return JudgeVerdict(correct=False, extracted_answer=None, reasoning="", error="judge timeout")


def _register_fixed_provider(name="fake-fixed", calls=None, dispatch=None):
    @registry.register_provider(name)
    class _FixedClient(BaseModelClient):
        protected_paths = ("model",)
        provider_name = name

        def build_request(self, prompt, max_output_tokens):
            return {"model": self.model, "prompt": prompt}

        async def _dispatch(self, request):
            if calls is not None:
                calls.append(request)
            if dispatch is not None:
                return await dispatch(request)
            return ModelResponse(
                text="<answer>6</answer>",
                tokens=TokenUsage(prompt_tokens=10, answer_tokens=5, reasoning_tokens=15,
                                  output_tokens=20, total_tokens=30),
                latency=0, model=self.model, finish_reason="stop",
            )

    return name


def _dataset(tmp):
    path = Path(tmp) / "ds.jsonl"
    path.write_text(
        json.dumps({"problem": "q1", "answer": "6", "id": "p1"}) + "\n" +
        json.dumps({"problem": "q2", "answer": "7", "id": "p2"}) + "\n",
        encoding="utf-8",
    )
    return str(path)


def _config(provider, dataset_path, **extra):
    base = dict(dataset_path=dataset_path, provider=provider, model="m",
                max_output_tokens=100, evaluator_type="math",
                judge=JudgeConfig(model="judge-m", base_url="https://judge/v1", api_key="jk"))
    base.update(extra)
    return BenchmarkConfig(**base)


# --------------------------------------------------------------------------- #
# RunCache unit behavior
# --------------------------------------------------------------------------- #
def test_cache_header_and_identity_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        cfg = _config("gemini", _dataset(tmp))
        RunCache.open(cache_path, cfg)
        header = json.loads(Path(cache_path).read_text().splitlines()[0])
        assert header["__ockbench_cache__"] is True
        assert header["identity"]["provider"] == "gemini"
        # Reopening with the SAME identity is fine.
        RunCache.open(cache_path, cfg)


def test_changed_identity_refused_naming_change():
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("gemini", ds, temperature=0.0))
        with pytest.raises(CacheIdentityMismatch) as exc:
            RunCache.open(cache_path, _config("gemini", ds, temperature=0.7))
        assert "generation" in str(exc.value)


def test_wall_clock_timeout_default_does_not_enter_identity():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config("gemini", _dataset(tmp))
        assert "wall_clock_timeout" not in compute_run_identity(cfg)["generation"]


def test_wall_clock_timeout_change_refused():
    # The optional wall-clock deadline changes retry/timeout outcomes, so a set
    # value is part of identity. The default None is omitted to preserve old
    # cache resume behavior.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("gemini", ds))
        with pytest.raises(CacheIdentityMismatch) as exc:
            RunCache.open(cache_path, _config("gemini", ds, wall_clock_timeout=300))
        msg = str(exc.value)
        assert "generation" in msg and "wall_clock_timeout" in msg


def test_invalid_config_does_not_poison_cache():
    # A construction error (protected override) with --cache must fail fast WITHOUT
    # writing an identity header, so the corrected rerun isn't blocked as a mismatch.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        bad = BenchmarkConfig(
            dataset_path=_dataset(tmp), provider="chat_completion", model="m",
            base_url="https://x/v1", api_key="k", max_output_tokens=100,
            evaluator_type="science",
            request_overrides={"set": {}, "unset": ["stream"]},  # protected -> rejected at client build
        )
        with pytest.raises(ValueError):
            BenchmarkRunner(bad, cache_path=cache_path).run()
        assert not Path(cache_path).exists()  # cache not poisoned


def test_execution_timeout_change_refused():
    # Code scoring is timeout-sensitive; execution_timeout is part of identity.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        base = dict(dataset_path=ds, provider="gemini", model="m",
                    max_output_tokens=100, evaluator_type="code")
        RunCache.open(cache_path, BenchmarkConfig(**base, execution_timeout=5))
        with pytest.raises(CacheIdentityMismatch) as exc:
            RunCache.open(cache_path, BenchmarkConfig(**base, execution_timeout=30))
        assert "evaluator_settings" in str(exc.value)


def test_judge_max_tokens_change_refused():
    # Judge generation settings change verdicts, so they are part of identity.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("gemini", ds,
                      judge=JudgeConfig(model="j", base_url="https://j/v1", api_key="k", max_tokens=500)))
        with pytest.raises(CacheIdentityMismatch) as exc:
            RunCache.open(cache_path, _config("gemini", ds,
                          judge=JudgeConfig(model="j", base_url="https://j/v1", api_key="k", max_tokens=1000)))
        assert "judge" in str(exc.value)


def test_judge_runtime_identity_change_refused(monkeypatch):
    # Code-level judge behavior (prompt/parser/default response_format/retry) can
    # change verdicts even when YAML config is unchanged, so it is part of cache
    # identity.
    import src.evaluators.judge as judge_mod

    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("gemini", ds))
        monkeypatch.setattr(judge_mod, "JUDGE_PARSER_VERSION", "changed")
        with pytest.raises(CacheIdentityMismatch) as exc:
            RunCache.open(cache_path, _config("gemini", ds))
        assert "judge" in str(exc.value)


def test_different_endpoint_refused():
    # Same provider/model/dataset but a different endpoint is a different run.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("chat_completion", ds,
                                          base_url="https://api.openai.com/v1", api_key="k"))
        with pytest.raises(CacheIdentityMismatch) as exc:
            RunCache.open(cache_path, _config("chat_completion", ds,
                                              base_url="https://relay.example.com/v1", api_key="k"))
        assert "base_url" in str(exc.value)


def test_prompt_template_change_refused(monkeypatch):
    # The prompt/template is part of run identity (AC-8 tuple); a template change
    # must refuse resume, naming prompt_template.
    import src.utils.prompt_formatter as pf
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("gemini", ds))
        monkeypatch.setattr(pf, "PROMPT_FORMATTER_VERSION", "999")
        with pytest.raises(CacheIdentityMismatch) as exc:
            RunCache.open(cache_path, _config("gemini", ds))
        assert "prompt_template" in str(exc.value)


def test_cache_header_has_no_credentials():
    # base_url credentials must never be written into the cache identity header.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("chat_completion", ds,
                                          base_url="https://user:s3cret@relay.example.com/v1", api_key="k"))
        raw = Path(cache_path).read_text()
        assert "s3cret" not in raw


def test_cache_header_masks_query_string_credentials():
    # Credentials in a base_url query string must not reach the cache header.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        ds = _dataset(tmp)
        RunCache.open(cache_path, _config("chat_completion", ds,
                                          base_url="https://relay/v1?api_key=QUERYSECRET", api_key="k"))
        raw = Path(cache_path).read_text()
        assert "QUERYSECRET" not in raw


def test_cache_header_redacts_override_secrets():
    # Secrets injected through request_overrides must not be written to the cache
    # identity header; non-secret override structure is retained.
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = str(Path(tmp) / "c.jsonl")
        cfg = BenchmarkConfig(
            dataset_path=_dataset(tmp), provider="gemini", model="m",
            max_output_tokens=100, evaluator_type="science",
            request_overrides={"set": {
                "headers.x-api-key": "OVERRIDE-SECRET",
                "extra_headers.Authorization": "Bearer BEARER-SECRET",
                "extra_body.reasoning.effort": "high",
            }, "unset": []},
        )
        RunCache.open(cache_path, cfg)
        raw = Path(cache_path).read_text()
        assert "OVERRIDE-SECRET" not in raw
        assert "BEARER-SECRET" not in raw
        assert "***MASKED***" in raw
        assert "reasoning" in raw  # non-secret override structure preserved


def test_judge_outage_recorded_as_error_and_retried(monkeypatch):
    # A judge outage must be recorded as an error (not a silent wrong answer) so
    # the cache does NOT mark the problem completed and a resume re-attempts it.
    provider = _register_fixed_provider("fake-fixed-outage")
    judge = _ErroringJudge()
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: judge)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            exp = BenchmarkRunner(_config(provider, ds), cache_path=cache_path).run()

            assert exp.summary.error_count == 2          # both judge-outages -> errors
            assert exp.summary.correct_count == 0
            assert all(r.error for r in exp.results)      # not silent correct=False
            assert exp.results[0].model_response == "<answer>6</answer>"  # model tokens/text preserved
            calls_first = judge.calls
            assert calls_first == 2

            # Resume: the outaged problems were not completed -> they are re-judged.
            BenchmarkRunner(_config(provider, ds), cache_path=cache_path).run()
            assert judge.calls > calls_first
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_empty_exception_recorded_as_error_and_retried(monkeypatch):
    # O4 regression: an exception whose str() is '' exhausting all retries must
    # still persist a non-empty cache error — '' is falsy, so the row would
    # otherwise be scored as an empty answer, read as completed on resume, and
    # never be re-attempted.
    dispatch_calls = []

    async def _raise(_request):
        raise EmptyStrError()

    _register_fixed_provider("fake-empty-exc", calls=dispatch_calls, dispatch=_raise)

    judge = _CountingJudge()
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: judge)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")

            exp1 = BenchmarkRunner(_config("fake-empty-exc", ds, max_retries=1),
                                   cache_path=cache_path).run()
            assert exp1.summary.error_count == 2
            assert judge.calls == 0  # never scored as an answer
            body = [json.loads(ln) for ln in Path(cache_path).read_text().splitlines()[1:] if ln.strip()]
            assert body and all(r["error"] for r in body)

            # Resume: error rows are not completed -> the model is re-attempted.
            calls_first = len(dispatch_calls)
            BenchmarkRunner(_config("fake-empty-exc", ds, max_retries=1),
                            cache_path=cache_path).run()
            assert len(dispatch_calls) > calls_first
    finally:
        registry._PROVIDER_REGISTRY.pop("fake-empty-exc", None)


def test_judge_outage_resume_rejudges_without_regenerating(monkeypatch):
    class _SwitchingJudge:
        def __init__(self):
            self.calls = 0
            self.fail = True

        async def score(self, *, question, ground_truth, candidate):
            self.calls += 1
            if self.fail:
                return JudgeVerdict(correct=False, extracted_answer=None, reasoning="", error="judge timeout")
            return JudgeVerdict(correct=(str(candidate) == str(ground_truth)),
                                extracted_answer=str(candidate), reasoning="ok")

    provider_calls = []
    provider = _register_fixed_provider("fake-fixed-rejudge-only", calls=provider_calls)
    judge = _SwitchingJudge()
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: judge)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)

            exp1 = BenchmarkRunner(cfg, cache_path=cache_path).run()
            assert exp1.summary.error_count == 2
            assert len(provider_calls) == 2
            assert judge.calls == 2

            judge.fail = False
            exp2 = BenchmarkRunner(cfg, cache_path=cache_path).run()
            assert len(provider_calls) == 2  # no second model-generation pass
            assert judge.calls == 4
            assert exp2.summary.error_count == 0
            assert exp2.summary.correct_count == 1
            assert exp2.summary.total_tokens == exp1.summary.total_tokens
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


# --------------------------------------------------------------------------- #
# End-to-end: resume skips & does not re-judge; results derive from cache
# --------------------------------------------------------------------------- #
def test_resume_skips_and_does_not_rejudge(monkeypatch):
    provider = _register_fixed_provider("fake-fixed-resume")
    judge = _CountingJudge()
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: judge)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            cfg = _config(provider, ds)

            exp1 = BenchmarkRunner(cfg, cache_path=cache_path).run()
            assert exp1.summary.total_problems == 2
            assert exp1.summary.correct_count == 1  # p1 answer 6 matches; p2 does not
            assert judge.calls == 2

            # Cache holds a header + the two outcomes.
            lines = [ln for ln in Path(cache_path).read_text().splitlines() if ln.strip()]
            assert json.loads(lines[0]).get("__ockbench_cache__") is True
            assert len(lines) == 3

            # Resume with the same identity: nothing to do, judge not re-invoked.
            exp2 = BenchmarkRunner(_config(provider, ds), cache_path=cache_path).run()
            assert judge.calls == 2
            assert exp2.summary.correct_count == 1

            # Results regenerated purely from the cache match the run's results.
            results, summary = aggregate_cache_file(cache_path)
            assert summary.correct_count == exp1.summary.correct_count
            assert summary.total_tokens == exp1.summary.total_tokens
            assert summary.ock_score == exp1.summary.ock_score
            assert len(results) == 2
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)


def test_results_are_pure_aggregation_of_cache(monkeypatch):
    provider = _register_fixed_provider("fake-fixed-agg")
    judge = _CountingJudge()
    monkeypatch.setattr(math_eval, "build_judge", lambda cfg: judge)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ds = _dataset(tmp)
            cache_path = str(Path(tmp) / "c.jsonl")
            exp = BenchmarkRunner(_config(provider, ds), cache_path=cache_path).run()
            # The runner's reported summary equals a fresh aggregation of the cache.
            _, summary = aggregate_cache_file(cache_path)
            assert summary.accuracy == exp.summary.accuracy
            assert summary.total_output_tokens == exp.summary.total_output_tokens
    finally:
        registry._PROVIDER_REGISTRY.pop(provider, None)
