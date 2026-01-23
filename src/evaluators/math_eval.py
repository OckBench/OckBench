"""
Math problem evaluator with answer extraction and comparison.
"""
import re
import logging
from typing import Any, Optional, Tuple


logger = logging.getLogger(__name__)


class MathEvaluator:
    """
    Evaluator for mathematical reasoning problems.
    
    Extracts answers from model responses using multiple patterns
    and compares with ground truth.
    """
    
    def __init__(self):
        """Initialize math evaluator."""
        self.extraction_patterns = [
            # Pattern 1: LaTeX boxed format - \boxed{answer}
            (r'\\boxed\{([^}]+)\}', 'boxed'),
            
            # Pattern 2: XML/HTML-style tags - <answer>value</answer>
            (r'<answer>([^<]+)</answer>', 'xml_tag'),
            
            # Pattern 3: Markdown code block with answer tag
            (r'```answer\s*\n([^`]+)\n```', 'code_block'),
            
            # Pattern 4: #### marker (common in GSM8K format)
            (r'####\s*([^\n]+)', 'gsm8k_marker'),
            
            # Pattern 5: "The answer is X" patterns
            (r'[Tt]he (?:final )?answer is[:\s]+([^\n.]+)', 'answer_is'),
            
            # Pattern 6: "Final answer: X"
            (r'[Ff]inal [Aa]nswer[:\s]+([^\n.]+)', 'final_answer'),
            
            # Pattern 7: Answer at end after "Answer:"
            (r'[Aa]nswer[:\s]+([^\n]+)$', 'answer_colon'),
            
            # Pattern 8: "The value/length/number... is X" at the end
            (r'(?:[Tt]he (?:value|length|number|distance|least|greatest)(?: [^.]*)? is|which is)\s+\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\$?[\s.]*\Z', 'value_is_at_end'),
            
            # Pattern 9: "= NUMBER" near the end (common in calculations like "m+n = 809")
            # This should come before last_number to catch final results
            # Using \Z instead of $ to match only at end of string (not end of line)
            # Handles optional LaTeX $ delimiters and closing brackets \] after the number
            (r'=\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\$?(?:\s|\.|\\\])*\Z', 'equals_at_end'),
            
            # Pattern 10: Last number in the response (fallback)
            (r'(?:^|\s)(-?\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})*(?:\.\d+)?)(?:\s|$)', 'last_number'),
        ]
    
    def extract_answer(self, response: str) -> Tuple[Optional[Any], str]:
        """
        Extract answer from model response using multiple patterns.
        
        Tries patterns in order of specificity, returns first match.
        
        Args:
            response: Model response text
        
        Returns:
            Tuple of (extracted_answer, extraction_method)
        """
        if not response or not response.strip():
            return None, 'empty_response'
        
        # Try each pattern in order
        for pattern, method_name in self.extraction_patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            
            if matches:
                # For last_number pattern, take the actual last match
                if method_name == 'last_number':
                    answer = matches[-1]
                else:
                    answer = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
                
                # Normalize the extracted answer
                normalized = self._normalize_answer(answer)
                if normalized is not None:
                    logger.debug(f"Extracted answer '{normalized}' using method '{method_name}'")
                    return normalized, method_name
        
        # No pattern matched
        logger.warning(f"Could not extract answer from response: {response[:100]}...")
        return None, 'no_match'
    
    def _normalize_answer(self, answer: str) -> Optional[Any]:
        """
        Normalize extracted answer for comparison.
        
        Args:
            answer: Raw extracted answer string
        
        Returns:
            Normalized answer (number, string, or None)
        """
        if not answer:
            return None
        
        # Strip whitespace and common punctuation
        answer = answer.strip().strip('.,;:!?')
        
        # Remove common prefixes/suffixes
        answer = re.sub(r'^(is|equals?|=)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip()
        
        if not answer:
            return None
        
        # Try to convert to number
        try:
            # Remove commas from numbers like "1,000"
            answer_no_commas = answer.replace(',', '')
            
            # Try integer first
            if '.' not in answer_no_commas:
                return int(answer_no_commas)
            else:
                # Try float
                return float(answer_no_commas)
        except ValueError:
            # Not a number, return as string (lowercased for comparison)
            return answer.lower().strip()
    
    def compare_answers(self, predicted: Any, ground_truth: Any) -> bool:
        """
        Compare predicted answer with ground truth.
        
        Args:
            predicted: Predicted answer (can be None)
            ground_truth: Ground truth answer
        
        Returns:
            bool: True if answers match
        """
        if predicted is None:
            return False
        
        # Normalize ground truth as well
        if isinstance(ground_truth, str):
            ground_truth = self._normalize_answer(ground_truth)
        
        # Type conversions for comparison
        # If one is int and other is float, compare as floats
        if isinstance(predicted, (int, float)) and isinstance(ground_truth, (int, float)):
            # Allow small floating point differences
            return abs(float(predicted) - float(ground_truth)) < 1e-6
        
        # If both are strings, compare case-insensitive
        if isinstance(predicted, str) and isinstance(ground_truth, str):
            return predicted.lower().strip() == ground_truth.lower().strip()
        
        # Try direct comparison
        try:
            return predicted == ground_truth
        except:
            return False
    
    def evaluate(self, response: str, ground_truth: Any):
        """Evaluate model response against ground truth."""
        from . import EvalResult
        extracted_answer, method = self.extract_answer(response)
        is_correct = self.compare_answers(extracted_answer, ground_truth)
        return EvalResult(is_correct=is_correct, extracted_answer=extracted_answer, extraction_method=method)


def get_evaluator(evaluator_type: str = "math", **kwargs):
    """
    Factory function to get appropriate evaluator.
    
    Args:
        evaluator_type: Type of evaluator ('math', 'code', etc.)
        **kwargs: Additional arguments for evaluator (e.g., timeout for code)
    
    Returns:
        Evaluator instance
    """
    if evaluator_type == "math":
        return MathEvaluator()
    elif evaluator_type == "code":
        from .code_eval import CodeEvaluator
        timeout = kwargs.get('timeout', 5)
        return CodeEvaluator(timeout=timeout)
    else:
        raise NotImplementedError(f"Evaluator type '{evaluator_type}' not implemented")

