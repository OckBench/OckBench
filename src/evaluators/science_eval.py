"""
Science problem evaluator for multiple-choice questions.

Handles GPQA-style questions where the answer is a single letter (A, B, C, D).
"""
import re
import logging
from typing import Any, Optional, Tuple


logger = logging.getLogger(__name__)


class ScienceEvaluator:
    """
    Evaluator for multiple-choice science problems.

    Extracts the selected answer (A, B, C, or D) from model responses
    and compares with ground truth.
    """

    def __init__(self):
        """Initialize science evaluator."""
        self.valid_choices = {'A', 'B', 'C', 'D'}

        # Patterns to extract multiple choice answers, ordered by specificity
        self.extraction_patterns = [
            # Pattern 0: <answer>X</answer> tags (explicit format requested in prompt)
            (r'<answer>\s*\(?([A-Da-d])\)?\s*</answer>', 'answer_tags'),

            # Pattern 1: "The answer is X" or "The answer is (X)"
            (r'[Tt]he answer is[:\s]*\(?([A-Da-d])\)?', 'answer_is'),

            # Pattern 2: "Answer: X" or "Answer: (X)"
            (r'[Aa]nswer[:\s]*\(?([A-Da-d])\)?', 'answer_colon'),

            # Pattern 3: "Final answer: X"
            (r'[Ff]inal [Aa]nswer[:\s]*\(?([A-Da-d])\)?', 'final_answer'),

            # Pattern 4: "I choose X" or "I select X"
            (r'I (?:choose|select|pick)[:\s]*\(?([A-Da-d])\)?', 'i_choose'),

            # Pattern 5: "Option X" or "Choice X" at the end
            (r'(?:[Oo]ption|[Cc]hoice)[:\s]*\(?([A-Da-d])\)?[.\s]*$', 'option_choice'),

            # Pattern 6: Boxed answer \boxed{X}
            (r'\\boxed\{([A-Da-d])\}', 'boxed'),

            # Pattern 7: Standalone letter at end of response (after reasoning)
            # Match letter that appears alone near the end, possibly in parens/brackets
            (r'(?:^|\n)\s*[\[\(]?([A-Da-d])[\]\)]?\.?\s*$', 'standalone_letter'),

            # Pattern 8: "X." or "(X)" or "[X]" appearing as the final answer indicator
            (r'(?:Therefore|Thus|Hence|So)[,:]?\s*(?:the answer is\s*)?[\[\(]?([A-Da-d])[\]\)]?', 'therefore'),
        ]

    def extract_answer(self, response: str) -> Tuple[Optional[str], str]:
        """
        Extract the multiple-choice answer from model response.

        Args:
            response: Model response text

        Returns:
            Tuple of (extracted_answer, extraction_method)
        """
        if not response or not response.strip():
            return None, 'empty_response'

        # Try each pattern in order
        for pattern, method_name in self.extraction_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)

            if matches:
                # Take the last match (final answer is usually at the end)
                answer = matches[-1].upper()
                if answer in self.valid_choices:
                    logger.debug(f"Extracted answer '{answer}' using method '{method_name}'")
                    return answer, method_name

        # Fallback: Look for any standalone letter A-D in the last 200 chars
        last_part = response[-200:] if len(response) > 200 else response
        letter_matches = re.findall(r'\b([A-Da-d])\b', last_part)
        if letter_matches:
            # Filter to valid choices
            valid_matches = [m.upper() for m in letter_matches if m.upper() in self.valid_choices]
            if valid_matches:
                logger.debug(f"Extracted answer '{valid_matches[-1]}' using fallback")
                return valid_matches[-1], 'fallback_letter'

        logger.warning(f"Could not extract answer from response: {response[-200:]}...")
        return None, 'no_match'

    def compare_answers(self, predicted: Optional[str], ground_truth: Any) -> bool:
        """
        Compare predicted answer with ground truth.

        Args:
            predicted: Predicted answer (should be A, B, C, or D)
            ground_truth: Ground truth answer

        Returns:
            bool: True if answers match
        """
        if predicted is None:
            return False

        # Normalize both to uppercase single letter
        pred_normalized = str(predicted).strip().upper()
        gt_normalized = str(ground_truth).strip().upper()

        # Handle various ground truth formats
        # Could be "A", "(A)", "A.", "Option A", etc.
        gt_match = re.search(r'([A-D])', gt_normalized)
        if gt_match:
            gt_normalized = gt_match.group(1)

        return pred_normalized == gt_normalized

    def evaluate(self, response: str, ground_truth: Any):
        """
        Evaluate model response against ground truth.

        Args:
            response: Model response text
            ground_truth: Expected answer (A, B, C, or D)

        Returns:
            EvalResult with correctness and extracted answer
        """
        from . import EvalResult

        extracted_answer, method = self.extract_answer(response)
        is_correct = self.compare_answers(extracted_answer, ground_truth)

        return EvalResult(
            is_correct=is_correct,
            extracted_answer=extracted_answer,
            extraction_method=method
        )
