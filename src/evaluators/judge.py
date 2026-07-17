"""LLM judge for math scoring (the default, required math scorer).

The judge is an inline, first-class scorer. It receives the *extracted* answer
block (not the raw transcript) plus the question and ground truth, and returns a
verdict. It is an OpenAI-compatible client; its SDK auto-retry is disabled and a
small bounded retry handles transient/parse failures.
"""
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence

from openai import AsyncOpenAI

from ..core.schemas import JudgeConfig
from ..utils.request_overrides import apply_request_overrides

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "math-grader-json-v1"
JUDGE_PARSER_VERSION = "json-recovery-v2"
DEFAULT_JUDGE_MAX_ATTEMPTS = 4
# Judge endpoints drop out in bursty bad windows lasting tens of seconds
# (transient zero-length responses; observed on both gateways). Retries only
# help if they (a) span longer than a bad window and (b) are jittered so
# concurrently-failing rows don't retry in lockstep and collide again.
DEFAULT_JUDGE_BACKOFF: Sequence[float] = (5.0, 15.0, 45.0)
DEFAULT_JUDGE_BACKOFF_JITTER = 3.0

# Sent with every judge request and hashed into the cache identity via
# runtime_identity() — one source, so changing a default here automatically
# blocks resume into caches scored under the old defaults.
JUDGE_REQUEST_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.0,
    "response_format": {"type": "json_object"},
}


def runtime_identity() -> Dict[str, Any]:
    """Identity of code-level judge behavior not otherwise present in config."""
    return {
        "prompt_version": JUDGE_PROMPT_VERSION,
        "parser_version": JUDGE_PARSER_VERSION,
        "request_defaults": JUDGE_REQUEST_DEFAULTS,
        "retry": {
            "max_attempts": DEFAULT_JUDGE_MAX_ATTEMPTS,
            "backoff": list(DEFAULT_JUDGE_BACKOFF),
            "jitter": DEFAULT_JUDGE_BACKOFF_JITTER,
        },
    }

JUDGE_PROMPT_TEMPLATE = """\
You are a math grader. Decide whether the student's answer matches the ground-truth answer.

## Problem
{question}

## Ground Truth Answer
{ground_truth}

## Student's Answer
{candidate}

## Instructions
1. The answer is correct if it matches the ground truth numerically or semantically (e.g. "1/2" == "0.5", "two" == "2").
2. Ignore minor formatting differences (e.g. "42" vs "42.0", "$100" vs "100", surrounding prose).
3. If the student's answer still contains reasoning, identify the final answer within it.

## Response Format
Respond with ONLY a JSON object in this exact format:
{{"correct": true/false, "extracted_answer": "the student's final answer", "reasoning": "brief explanation"}}
"""


@dataclass
class JudgeVerdict:
    """A single judge decision over one candidate answer."""
    correct: bool
    extracted_answer: Optional[str] = None
    reasoning: str = ""
    error: Optional[str] = None


class Judge(Protocol):
    """Scores a candidate answer against ground truth. Mockable in tests."""

    async def score(self, *, question: str, ground_truth: Any, candidate: str) -> JudgeVerdict:
        ...


def _coerce_correct(value: Any) -> bool:
    """Strictly interpret the judge's ``correct`` field.

    Truthiness is unsafe here: a judge that emits a string boolean like
    ``"correct": "false"`` would otherwise be read as True (non-empty string),
    marking a wrong math answer correct. Only a real ``True``, a recognized
    affirmative string (``true``/``yes``/``1``), or numeric ``1`` counts as
    correct; anything else — including ``"false"`` — is False.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def parse_json_judgment(content: str) -> Dict[str, Any]:
    """Parse the judge JSON even if a thinking model emits surrounding text."""
    content = content.strip()

    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 3:
            fenced = parts[1].strip()
            if fenced.startswith("json"):
                fenced = fenced[4:].strip()
            content = fenced

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    matches = re.findall(r"\{.*?\}", content, flags=re.DOTALL)
    for match in reversed(matches):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Truncated-JSON recovery: with response_format=json_object a verbose judge can
    # hit max_tokens mid-"reasoning", leaving the object unclosed. The verdict
    # fields precede reasoning in our format, so recover them from the prefix
    # instead of failing (and burning retries) on an otherwise-valid verdict.
    if content.lstrip().startswith("{"):
        m = re.search(r'"correct"\s*:\s*("?(?:true|false)"?)', content, flags=re.IGNORECASE)
        if m:
            out: Dict[str, Any] = {"correct": m.group(1).strip('"')}
            m2 = re.search(r'"extracted_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
            if m2:
                try:
                    out["extracted_answer"] = json.loads('"' + m2.group(1) + '"')
                except json.JSONDecodeError:
                    out["extracted_answer"] = m2.group(1)
            m3 = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', content)
            if m3:
                out["reasoning"] = m3.group(1)[:500] + " [truncated]"
            return out

    # NOTE: the trailing pos arg renders as "(char 0)" regardless of content —
    # include the content head so refusals/prose are visible in the error itself.
    raise json.JSONDecodeError(
        f"No JSON object found in judge response (head: {content[:120]!r})", content, 0,
    )


class LLMJudge:
    """Default math judge backed by an OpenAI-compatible endpoint."""

    def __init__(self, config: JudgeConfig, max_attempts: int = DEFAULT_JUDGE_MAX_ATTEMPTS,
                 backoff: Optional[Sequence[float]] = None):
        self.config = config
        self.max_attempts = max_attempts
        self.backoff = tuple(backoff) if backoff is not None else DEFAULT_JUDGE_BACKOFF
        # Force JSON output when the endpoint supports it: contest-style questions
        # embedded in the judge prompt can hijack the judge into prose/solving mode
        # (observed: AMO items -> non-JSON responses across judge models). Disabled
        # automatically on endpoints that reject response_format.
        self._json_mode = True

        client_kwargs: Dict[str, Any] = {"max_retries": 0}  # we own retry
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
            if "api_key" not in client_kwargs:
                client_kwargs["api_key"] = "dummy-key"
        self.client = AsyncOpenAI(**client_kwargs)

    def _build_request(self, prompt: str) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": JUDGE_REQUEST_DEFAULTS["temperature"],
            "max_tokens": self.config.max_tokens,
        }
        if self._json_mode:
            request["response_format"] = JUDGE_REQUEST_DEFAULTS["response_format"]
        return apply_request_overrides(request, self.config.request_overrides, {})

    async def score(self, *, question: str, ground_truth: Any, candidate: str) -> JudgeVerdict:
        # Fail closed on a blank candidate instead of asking the LLM to grade
        # nothing: the prompt embeds the ground truth, so a graded blank can
        # come back as a hallucinated match. Callers normally short-circuit
        # empty responses before reaching the judge; this is the last line of
        # defense for any path that doesn't.
        if not candidate or not candidate.strip():
            return JudgeVerdict(
                correct=False, extracted_answer=None,
                reasoning="empty candidate: judged incorrect without an LLM call",
            )

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, ground_truth=ground_truth, candidate=candidate,
        )
        last_error: Optional[str] = None
        for attempt in range(self.max_attempts):
            request = self._build_request(prompt)
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**request),
                    timeout=self.config.timeout,
                )
                content = (response.choices[0].message.content or "").strip()
                verdict = parse_json_judgment(content)
                return JudgeVerdict(
                    correct=_coerce_correct(verdict.get("correct", False)),
                    extracted_answer=(str(verdict["extracted_answer"])
                                      if verdict.get("extracted_answer") is not None else None),
                    reasoning=str(verdict.get("reasoning", "")),
                )
            except (json.JSONDecodeError, asyncio.TimeoutError) as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"judge attempt {attempt + 1}/{self.max_attempts} failed: {last_error}")
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                if self._json_mode and "response_format" in last_error.lower():
                    self._json_mode = False
                    logger.warning("judge endpoint rejects response_format; disabling JSON mode")
                logger.warning(f"judge attempt {attempt + 1}/{self.max_attempts} error: {last_error}")
            if attempt < self.max_attempts - 1:
                wait = self.backoff[min(attempt, len(self.backoff) - 1)]
                await asyncio.sleep(wait + random.uniform(0, DEFAULT_JUDGE_BACKOFF_JITTER))

        return JudgeVerdict(correct=False, extracted_answer=None, reasoning="", error=last_error)


def build_judge(config: JudgeConfig) -> LLMJudge:
    return LLMJudge(config)
