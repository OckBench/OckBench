"""
Code evaluation module (placeholder for future implementation).

Future work:
- Execute code in sandboxed environment
- Compare outputs with expected results
- Support for MBPP, HumanEval, etc.
"""


class CodeEvaluator:
    """
    Placeholder for code execution evaluator.
    
    To implement:
    1. Extract code from model response
    2. Execute in sandboxed environment (docker, etc.)
    3. Compare outputs with test cases
    4. Handle timeouts and errors
    """
    
    def __init__(self):
        raise NotImplementedError(
            "CodeEvaluator is not yet implemented. "
            "Use MathEvaluator for math problems."
        )
    
    def evaluate(self, response: str, test_cases: list) -> bool:
        """Evaluate code response against test cases."""
        raise NotImplementedError()

