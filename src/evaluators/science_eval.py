"""Science evaluator for multiple-choice questions (A/B/C/D)."""
import logging
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ScienceEvaluator:
    def __init__(self):
        self.valid_choices = {'A', 'B', 'C', 'D'}

        self.extraction_patterns = [
            (r'<answer>\s*\(?([A-Da-d])\)?\s*</answer>', 'answer_tags'),
            (r'[Tt]he answer is[:\s]*\(?([A-Da-d])\)?', 'answer_is'),
            (r'[Aa]nswer[:\s]*\(?([A-Da-d])\)?', 'answer_colon'),
            (r'[Ff]inal [Aa]nswer[:\s]*\(?([A-Da-d])\)?', 'final_answer'),
            (r'I (?:choose|select|pick)[:\s]*\(?([A-Da-d])\)?', 'i_choose'),
            (r'(?:[Oo]ption|[Cc]hoice)[:\s]*\(?([A-Da-d])\)?[.\s]*$', 'option_choice'),
            (r'\\boxed\{([A-Da-d])\}', 'boxed'),
            (r'(?:^|\n)\s*[\[\(]?([A-Da-d])[\]\)]?\.?\s*$', 'standalone_letter'),
            (r'(?:Therefore|Thus|Hence|So)[,:]?\s*(?:the answer is\s*)?[\[\(]?([A-Da-d])[\]\)]?', 'therefore'),
        ]

    def extract_answer(self, response: str) -> Tuple[Optional[str], str]:
        if not response or not response.strip():
            return None, 'empty_response'

        for pattern, method_name in self.extraction_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            if matches:
                answer = matches[-1].upper()
                if answer in self.valid_choices:
                    return answer, method_name

        # Fallback: any standalone A-D in last 200 chars
        last_part = response[-200:] if len(response) > 200 else response
        letter_matches = re.findall(r'\b([A-Da-d])\b', last_part)
        if letter_matches:
            valid_matches = [m.upper() for m in letter_matches if m.upper() in self.valid_choices]
            if valid_matches:
                return valid_matches[-1], 'fallback_letter'

        return None, 'no_match'

    def compare_answers(self, predicted: Optional[str], ground_truth: Any) -> bool:
        if predicted is None:
            return False
        pred_normalized = str(predicted).strip().upper()
        gt_normalized = str(ground_truth).strip().upper()
        gt_match = re.search(r'([A-D])', gt_normalized)
        if gt_match:
            gt_normalized = gt_match.group(1)
        return pred_normalized == gt_normalized

    def evaluate(self, response: str, ground_truth: Any):
        from . import EvalResult
        extracted_answer, method = self.extract_answer(response)
        is_correct = self.compare_answers(extracted_answer, ground_truth)
        return EvalResult(is_correct=is_correct, extracted_answer=extracted_answer, extraction_method=method)
