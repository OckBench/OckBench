"""LLM judge for math scoring (the default, required math scorer).

Ported from the former post-hoc ``scripts/llm_eval.py`` and made an inline,
first-class scorer. The judge receives the *extracted* answer block (not the raw
transcript) plus the question and ground truth, and returns a verdict. It is an
OpenAI-compatible client; its SDK auto-retry is disabled and a small bounded
retry handles transient/parse failures.
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from openai import AsyncOpenAI

from ..core.schemas import JudgeConfig
from ..utils.request_overrides import apply_request_overrides

logger = logging.getLogger(__name__)

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

    raise json.JSONDecodeError("No JSON object found in judge response", content, 0)


class LLMJudge:
    """Default math judge backed by an OpenAI-compatible endpoint."""

    def __init__(self, config: JudgeConfig, max_attempts: int = 3):
        self.config = config
        self.max_attempts = max_attempts

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
            "temperature": 0.0,
            "max_tokens": self.config.max_tokens,
        }
        return apply_request_overrides(request, self.config.request_overrides, {})

    async def score(self, *, question: str, ground_truth: Any, candidate: str) -> JudgeVerdict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, ground_truth=ground_truth, candidate=candidate,
        )
        request = self._build_request(prompt)

        last_error: Optional[str] = None
        for attempt in range(self.max_attempts):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**request),
                    timeout=self.config.timeout,
                )
                content = (response.choices[0].message.content or "").strip()
                verdict = parse_json_judgment(content)
                return JudgeVerdict(
                    correct=bool(verdict.get("correct", False)),
                    extracted_answer=(str(verdict["extracted_answer"])
                                      if verdict.get("extracted_answer") is not None else None),
                    reasoning=str(verdict.get("reasoning", "")),
                )
            except (json.JSONDecodeError, asyncio.TimeoutError) as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"judge attempt {attempt + 1}/{self.max_attempts} failed: {last_error}")
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                logger.warning(f"judge attempt {attempt + 1}/{self.max_attempts} error: {last_error}")
            if attempt < self.max_attempts - 1:
                await asyncio.sleep(2 ** attempt)

        return JudgeVerdict(correct=False, extracted_answer=None, reasoning="", error=last_error)


def build_judge(config: JudgeConfig) -> LLMJudge:
    return LLMJudge(config)
