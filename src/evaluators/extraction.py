"""Answer-block extraction, kept separate from scoring.

Scoring (is this answer correct?) and extraction (what did the model put forward
as its answer?) are different concerns. This module owns extraction of the
designated answer block that task prompts instruct models to emit, so a scorer
can judge the isolated block instead of re-parsing a whole transcript.

For math the block is ``<answer>...</answer>``; the extracted content is handed
to the LLM judge. The regex here is the *only* regex left in the math path — it
no longer decides correctness, it only isolates the block.
"""
import re
from typing import Optional, Tuple

# Last <answer>...</answer> block, across newlines. Models are instructed to put
# their final answer at the very end, so the last block is the operative one.
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer_block(text: str) -> Tuple[Optional[str], str]:
    """Return the content of the final ``<answer>`` block and the method tag.

    Returns ``(content, "answer_block")`` when a block is present (content
    stripped of surrounding whitespace), ``(None, "empty_response")`` for empty
    input, and ``(None, "no_answer_block")`` when no block is found.
    """
    if not text or not text.strip():
        return None, "empty_response"

    matches = _ANSWER_BLOCK.findall(text)
    if matches:
        content = matches[-1].strip()
        if content:
            return content, "answer_block"

    return None, "no_answer_block"
