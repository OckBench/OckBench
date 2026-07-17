"""Offline unit tests for the token-usage normalization seam.

These tests exercise ``src.utils.usage_normalizer`` in isolation. They import
only the seam and the public schema; no provider SDK, no httpx client, and no
network or API key is touched. The import boundary is asserted explicitly by
``test_import_boundary_no_provider_or_http_modules``.
"""
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from src.core.schemas import TokenUsage
from src.utils.usage_normalizer import (
    NormalizedUsage,
    extract_gemini_usage,
    extract_openai_usage,
    extract_responses_usage,
    normalize_anthropic_usage,
    to_token_usage,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Local fakes (no SDK): attribute-style usage objects
# --------------------------------------------------------------------------- #
class _Details:
    def __init__(self, reasoning_tokens):
        self.reasoning_tokens = reasoning_tokens


class _OpenAIUsage:
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0,
                 reasoning_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        if reasoning_tokens is not None:
            self.completion_tokens_details = _Details(reasoning_tokens)


class _GeminiMetadata:
    """usage_metadata fake exposing only the attributes passed in."""

    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


# --------------------------------------------------------------------------- #
# AC-1: NormalizedUsage + to_token_usage
# --------------------------------------------------------------------------- #
def test_to_token_usage_passes_through_all_counts():
    n = NormalizedUsage(prompt_tokens=10, reasoning_tokens=20, answer_tokens=30,
                        output_tokens=50, total_tokens=60)
    tu = to_token_usage(n)
    assert isinstance(tu, TokenUsage)
    assert tu.prompt_tokens == 10
    assert tu.reasoning_tokens == 20
    assert tu.answer_tokens == 30
    assert tu.output_tokens == 50
    assert tu.total_tokens == 60


def test_to_token_usage_lets_schema_autofill_when_zero():
    n = NormalizedUsage(prompt_tokens=10, reasoning_tokens=20, answer_tokens=30)
    tu = to_token_usage(n)
    assert tu.output_tokens == 50  # reasoning + answer
    assert tu.total_tokens == 60   # prompt + answer + reasoning


def test_raw_usage_payload_is_absent():
    with pytest.raises(ImportError):
        from src.utils.usage_normalizer import RawUsagePayload  # noqa: F401


def test_normalized_usage_field_set_is_fixed():
    with pytest.raises(TypeError):
        NormalizedUsage(unknown_field=1)


# --------------------------------------------------------------------------- #
# AC-2: OpenAI chat-completions extractor
# --------------------------------------------------------------------------- #
def test_extract_openai_usage_basic():
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=5, completion_tokens=100,
                                          total_tokens=105, reasoning_tokens=30))
    assert n.answer_tokens == 70
    assert n.reasoning_tokens == 30
    assert n.output_tokens == 100
    assert n.prompt_tokens == 5
    assert n.total_tokens == 105


def test_extract_openai_usage_small():
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=3, completion_tokens=11,
                                          total_tokens=14, reasoning_tokens=4))
    assert n.answer_tokens == 7


def test_extract_openai_usage_without_details():
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=2, completion_tokens=9,
                                          total_tokens=11))
    assert n.reasoning_tokens == 0
    assert n.answer_tokens == 9  # answer == completion when no reasoning detail
    assert n.unattributed_tokens == 0


def test_extract_openai_usage_hidden_total_gap_folds_into_reasoning():
    # Regression (InfiniAI gemini-3.5-flash streaming): the relay bills hidden
    # thinking only in the total — prompt=15, visible completion=3, total=163,
    # no reasoning details. The 145-token gap is hidden output: it must land in
    # output/reasoning (not vanish), with the provider's original attribution
    # preserved for audit.
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=15, completion_tokens=3,
                                          total_tokens=163))
    assert n.answer_tokens == 3          # visible answer stays provider-exact
    assert n.reasoning_tokens == 145
    assert n.output_tokens == 148        # total - prompt
    assert n.total_tokens == 163
    assert n.unattributed_tokens == 145
    assert n.answer_tokens_estimated is False
    assert n.prompt_tokens + n.output_tokens == n.total_tokens
    assert n.answer_tokens + n.reasoning_tokens == n.output_tokens


def test_extract_openai_usage_hidden_gap_with_empty_answer():
    # The dangerous shape: finish_reason=length with no visible answer, all
    # billed thinking hidden in the total. Normalized output must be the gap,
    # not zero — a zero output would classify the row as a retryable broken
    # relay and re-burn the whole budget on resume.
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=15, completion_tokens=0,
                                          total_tokens=163))
    assert n.output_tokens == 148
    assert n.reasoning_tokens == 148
    assert n.answer_tokens == 0
    assert n.unattributed_tokens == 148


def test_extract_openai_usage_gap_stacks_on_reported_reasoning():
    # Reasoning details and a hidden-total gap can coexist; the gap adds to the
    # reported reasoning while the visible answer split stays exact.
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=10, completion_tokens=20,
                                          total_tokens=50, reasoning_tokens=5))
    assert n.answer_tokens == 15         # completion - reported reasoning
    assert n.reasoning_tokens == 25      # reported 5 + gap 20
    assert n.output_tokens == 40
    assert n.unattributed_tokens == 20


def test_extract_openai_usage_negative_or_missing_total_stays_literal():
    # A relay total smaller than prompt+completion is not the hidden-output
    # signature; counts stay provider-literal. Same when the total is absent.
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=10, completion_tokens=20,
                                          total_tokens=25))
    assert n.output_tokens == 20
    assert n.unattributed_tokens == 0
    m = extract_openai_usage(_OpenAIUsage(prompt_tokens=10, completion_tokens=20))
    assert m.output_tokens == 20
    assert m.unattributed_tokens == 0


def test_unattributed_tokens_survive_into_token_usage():
    n = extract_openai_usage(_OpenAIUsage(prompt_tokens=15, completion_tokens=3,
                                          total_tokens=163))
    assert to_token_usage(n).unattributed_tokens == 145


# --------------------------------------------------------------------------- #
# AC-3: Responses API extractor
# --------------------------------------------------------------------------- #
def test_extract_responses_usage_basic():
    n = extract_responses_usage({
        "input_tokens": 10, "output_tokens": 80, "total_tokens": 90,
        "output_tokens_details": {"reasoning_tokens": 25},
    })
    assert n.prompt_tokens == 10
    assert n.output_tokens == 80
    assert n.reasoning_tokens == 25
    assert n.answer_tokens == 55
    assert n.total_tokens == 90


def test_extract_responses_usage_without_details():
    n = extract_responses_usage({"input_tokens": 4, "output_tokens": 12,
                                "total_tokens": 16})
    assert n.reasoning_tokens == 0
    assert n.answer_tokens == 12


# --------------------------------------------------------------------------- #
# AC-4: Gemini extractor with fallback chains
# --------------------------------------------------------------------------- #
def test_extract_gemini_usage_primary_fields():
    n = extract_gemini_usage(_GeminiMetadata(
        prompt_token_count=10, candidates_token_count=40,
        total_token_count=65, thoughts_token_count=15,
    ))
    assert n.prompt_tokens == 10
    assert n.answer_tokens == 40
    assert n.reasoning_tokens == 15
    assert n.output_tokens == 55  # reasoning + answer
    assert n.total_tokens == 65


def test_extract_gemini_usage_fallback_names_parse():
    # Only fallback names present; primaries absent must NOT zero everything.
    n = extract_gemini_usage(_GeminiMetadata(prompt_tokens=7, output_tokens=13))
    assert n.prompt_tokens == 7
    assert n.answer_tokens == 13
    assert n.output_tokens == 13  # reasoning 0 + answer 13
    assert n.total_tokens == 20   # computed: 7 + 13 + 0


def test_extract_gemini_usage_none_is_all_zero():
    n = extract_gemini_usage(None)
    assert (n.prompt_tokens, n.answer_tokens, n.reasoning_tokens,
            n.output_tokens, n.total_tokens) == (0, 0, 0, 0, 0)


# --------------------------------------------------------------------------- #
# AC-5: async Anthropic normalization with injected callback
# --------------------------------------------------------------------------- #
def _make_counter(return_value, calls, estimated=False):
    async def counter(text):
        calls.append(text)
        return return_value, estimated
    return counter


def test_normalize_anthropic_usage_basic():
    calls = []
    n = asyncio.run(normalize_anthropic_usage(
        prompt_tokens=5, output_tokens=50, final_text="hello",
        count_tokens=_make_counter(30, calls),
    ))
    assert n.answer_tokens == 30
    assert n.reasoning_tokens == 20
    assert n.output_tokens == 50
    assert n.prompt_tokens == 5
    assert n.total_tokens == 55
    assert n.answer_tokens_estimated is False
    assert calls == ["hello"]


def test_normalize_anthropic_usage_marks_estimated_split():
    # When the injected counter reports an estimate (count_tokens endpoint
    # unavailable), the flag must survive into TokenUsage for per-row audit.
    n = asyncio.run(normalize_anthropic_usage(
        prompt_tokens=5, output_tokens=50, final_text="hello",
        count_tokens=_make_counter(30, [], estimated=True),
    ))
    assert n.answer_tokens == 30
    assert n.answer_tokens_estimated is True
    assert to_token_usage(n).answer_tokens_estimated is True


def test_normalize_anthropic_usage_clamps_answer_to_output():
    calls = []
    n = asyncio.run(normalize_anthropic_usage(
        prompt_tokens=5, output_tokens=50, final_text="hello",
        count_tokens=_make_counter(60, calls),
    ))
    assert n.answer_tokens == 50  # clamped to output
    assert n.reasoning_tokens == 0


def test_normalize_anthropic_usage_empty_text_skips_callback():
    calls = []
    n = asyncio.run(normalize_anthropic_usage(
        prompt_tokens=5, output_tokens=40, final_text="",
        count_tokens=_make_counter(999, calls),
    ))
    assert n.answer_tokens == 0
    assert n.reasoning_tokens == 40  # all output is reasoning
    assert calls == []  # callback never awaited


def test_no_sync_anthropic_answer_extractor_exists():
    import src.utils.usage_normalizer as seam
    # The only Anthropic path is the async/callback form. There is no synchronous
    # extractor that derives the answer split from a payload.
    assert not hasattr(seam, "extract_anthropic_usage")


# --------------------------------------------------------------------------- #
# AC-7.1 / AC-7.2: cache logging preserved, cache not surfaced
# --------------------------------------------------------------------------- #
def test_anthropic_cache_logging_preserved(caplog):
    with caplog.at_level(logging.INFO, logger="src.utils.usage_normalizer"):
        asyncio.run(normalize_anthropic_usage(
            prompt_tokens=10, output_tokens=20, final_text="hi",
            count_tokens=_make_counter(5, []),
            cache_metrics={"cache_creation_tokens": 100, "cache_read_tokens": 0,
                           "cache_ephemeral_5m": 40, "cache_ephemeral_1h": 60},
        ))
    assert any("cache: creation=100" in r.message for r in caplog.records)


def test_no_cache_log_when_no_cache_tokens(caplog):
    with caplog.at_level(logging.INFO, logger="src.utils.usage_normalizer"):
        asyncio.run(normalize_anthropic_usage(
            prompt_tokens=10, output_tokens=20, final_text="hi",
            count_tokens=_make_counter(5, []),
        ))
    assert not any("cache:" in r.message for r in caplog.records)


def test_cache_tokens_not_surfaced_in_token_usage():
    n = asyncio.run(normalize_anthropic_usage(
        prompt_tokens=10, output_tokens=20, final_text="hi",
        count_tokens=_make_counter(5, []),
        cache_metrics={"cache_read_tokens": 500},
    ))
    tu = to_token_usage(n)
    # prompt_tokens stays input-only; cache is excluded from the public schema.
    assert tu.prompt_tokens == 10
    assert tu.output_tokens == 20
    # TokenUsage exposes no cache fields at all.
    assert set(TokenUsage.model_fields) == {
        "prompt_tokens", "answer_tokens", "reasoning_tokens",
        "output_tokens", "total_tokens", "answer_tokens_estimated",
        "unattributed_tokens",
    }


# --------------------------------------------------------------------------- #
# AC-8: edge cases + import-boundary
# --------------------------------------------------------------------------- #
def test_missing_openai_usage_attrs_all_zero():
    class _Empty:
        pass
    n = extract_openai_usage(_Empty())
    assert (n.prompt_tokens, n.answer_tokens, n.reasoning_tokens,
            n.output_tokens, n.total_tokens) == (0, 0, 0, 0, 0)


def test_missing_responses_usage_all_zero():
    assert extract_responses_usage(None) == NormalizedUsage()
    assert extract_responses_usage({}) == NormalizedUsage()


def test_reasoning_greater_than_output_repaired_openai_without_text():
    n = extract_openai_usage(_OpenAIUsage(completion_tokens=10, reasoning_tokens=30))
    assert n.answer_tokens == 0
    assert n.reasoning_tokens == 10
    assert n.answer_tokens + n.reasoning_tokens == n.output_tokens


def test_reasoning_greater_than_output_repaired_openai_with_text():
    n = extract_openai_usage(
        _OpenAIUsage(completion_tokens=10, reasoning_tokens=30),
        final_text="12345678",
    )
    assert n.answer_tokens == 2
    assert n.reasoning_tokens == 8
    assert n.answer_tokens + n.reasoning_tokens == n.output_tokens
    # A repaired impossible split is an estimate; a possible one is not.
    assert n.answer_tokens_estimated is True
    ok = extract_openai_usage(_OpenAIUsage(completion_tokens=10, reasoning_tokens=3))
    assert ok.answer_tokens_estimated is False


def test_reasoning_greater_than_output_repaired_responses():
    n = extract_responses_usage({"input_tokens": 1, "output_tokens": 10,
                                "output_tokens_details": {"reasoning_tokens": 30}})
    assert n.answer_tokens == 0
    assert n.reasoning_tokens == 10
    assert n.answer_tokens + n.reasoning_tokens == n.output_tokens


def test_gemini_primary_wins_over_fallback():
    # Both primary and fallback present: primary must win.
    n = extract_gemini_usage(_GeminiMetadata(
        prompt_token_count=10, prompt_tokens=999,
        candidates_token_count=20, output_tokens=999,
    ))
    assert n.prompt_tokens == 10
    assert n.answer_tokens == 20


def test_import_boundary_no_provider_or_http_modules():
    code = (
        "import sys\n"
        "import src.utils.usage_normalizer  # noqa\n"
        "forbidden = ('httpx', 'openai', 'anthropic', 'google', "
        "'google.genai', 'google.generativeai')\n"
        "leaked = [m for m in forbidden if m in sys.modules]\n"
        "assert not leaked, leaked\n"
        "print('CLEAN')\n"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
