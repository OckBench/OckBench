"""Code evaluator: extract, execute in subprocess with timeout, validate tests.

Scoring semantics are unchanged (golden-tested): the same extraction patterns,
the same subprocess test harness, and the same pass/fail accounting. Only the
call surface is adapted to the async evaluator interface; test cases come from
``problem.metadata['test_cases']``.
"""
import logging
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple

from .base import EvalResult, Evaluator, register_evaluator

logger = logging.getLogger(__name__)


class CodeEvaluator(Evaluator):
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.extraction_patterns = [
            (r'<solution>\s*(.*?)\s*</solution>', 'solution_tags', re.DOTALL),
            (r'```python\s*\n(.*?)\n```', 'markdown_python', re.DOTALL),
            (r'```\s*\n(.*?)\n```', 'markdown_generic', re.DOTALL),
            (r'(def\s+\w+\s*\([^)]*\):[^\n]*(?:\n(?:    |\t).*)*)', 'function_def', re.MULTILINE),
            (r'(class\s+\w+.*?:\s*\n(?:(?:    |\t).*\n)*)', 'class_def', re.MULTILINE),
        ]

    def _clean_code(self, code: str) -> str:
        """Remove prose text before actual code starts."""
        lines = code.split('\n')
        cleaned_lines = []
        in_code = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```'):
                continue
            if not in_code and not stripped:
                continue
            if not in_code:
                if re.match(r'^(class |def |import |from |@|\w+\s*=)', stripped):
                    in_code = True
                    cleaned_lines.append(line)
                continue
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines).strip()

    def extract_code(self, response: str) -> Tuple[Optional[str], str]:
        if not response or not response.strip():
            return None, 'empty_response'

        for pattern, method_name, flags in self.extraction_patterns:
            matches = re.findall(pattern, response, flags)
            if matches:
                code = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
                code = self._clean_code(code.strip())
                if code:
                    return code, method_name

        # Fallback: find function definitions
        func_pattern = r'def\s+\w+\s*\([^)]*\):'
        if re.search(func_pattern, response):
            lines = response.split('\n')
            code_lines = []
            in_function = False

            for line in lines:
                if re.match(r'^\s*def\s+\w+', line):
                    in_function = True
                    code_lines.append(line)
                elif in_function:
                    if line.strip() and not line.startswith((' ', '\t')):
                        break
                    code_lines.append(line)

            if code_lines:
                return '\n'.join(code_lines).strip(), 'fallback_extraction'

        return None, 'no_match'

    def _create_test_script(self, code: str, test_cases: List[str]) -> str:
        common_imports = [
            "from typing import List, Dict, Optional, Any, Tuple, Set",
            "from collections import defaultdict, Counter, deque",
            "from functools import lru_cache, reduce",
            "from itertools import permutations, combinations, product",
            "from heapq import heappush, heappop, heapify",
            "from bisect import bisect_left, bisect_right",
            "import math",
            "import sys",
            "",
        ]

        script_parts = [
            *common_imports,
            code,
            "",
            "def run_tests():",
            "    passed = 0",
            "    failed = 0",
            "    total = 0",
            ""
        ]

        for i, test in enumerate(test_cases):
            script_parts.extend([
                "    total += 1",
                "    try:",
                f"        {test}",
                "        passed += 1",
                "    except AssertionError as e:",
                "        failed += 1",
                f"        print('FAILED Test {i + 1}: ' + {repr(test)})",
                "    except Exception as e:",
                "        failed += 1",
                f"        print('ERROR Test {i + 1}: ' + type(e).__name__ + ': ' + str(e))",
                ""
            ])

        script_parts.extend([
            "    return passed, failed, total",
            "",
            "if __name__ == '__main__':",
            "    passed, failed, total = run_tests()",
            "    print(f'RESULTS: {passed}/{total} tests passed')",
            "    exit(0 if failed == 0 else 1)",
        ])

        return '\n'.join(script_parts)

    def execute_code(self, code: str, test_cases: List[str]) -> Tuple[bool, int, int, Optional[str]]:
        """Execute code with test cases in subprocess. Returns (all_passed, passed, total, error)."""
        if not code:
            return False, 0, len(test_cases), "No code to execute"
        if not test_cases:
            return False, 0, 0, "No test cases provided"

        script = self._create_test_script(code, test_cases)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(script)
            temp_file = f.name

        try:
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True, text=True, timeout=self.timeout,
            )

            tests_passed = 0
            tests_total = len(test_cases)
            error_message = None

            match = re.search(r'RESULTS: (\d+)/(\d+) tests passed', result.stdout)
            if match:
                tests_passed = int(match.group(1))
                tests_total = int(match.group(2))

            if result.returncode != 0:
                error_lines = []
                for line in result.stdout.split('\n'):
                    if line.startswith('FAILED') or line.startswith('ERROR'):
                        error_lines.append(line)
                if result.stderr:
                    stderr_lines = result.stderr.strip().split('\n')
                    error_lines.extend(stderr_lines[-3:])
                error_message = '\n'.join(error_lines) if error_lines else "Execution failed"

            all_passed = (tests_passed == tests_total and result.returncode == 0)
            return all_passed, tests_passed, tests_total, error_message

        except subprocess.TimeoutExpired:
            return False, 0, len(test_cases), f"Execution timeout after {self.timeout}s"
        except Exception as e:
            return False, 0, len(test_cases), f"{type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    async def evaluate(self, problem, response: str) -> EvalResult:
        test_cases = (problem.metadata or {}).get('test_cases', [])
        extracted_code, extraction_method = self.extract_code(response)

        if not extracted_code:
            return EvalResult(
                is_correct=False, extracted_answer=None, extraction_method=extraction_method,
                tests_passed=0, tests_total=len(test_cases), execution_error="Failed to extract code",
            )

        all_passed, tests_passed, tests_total, error_message = self.execute_code(extracted_code, test_cases)
        return EvalResult(
            is_correct=all_passed, extracted_answer=extracted_code, extraction_method=extraction_method,
            tests_passed=tests_passed, tests_total=tests_total, execution_error=error_message,
        )


@register_evaluator("code")
def _build_code_evaluator(config) -> CodeEvaluator:
    return CodeEvaluator(timeout=config.execution_timeout)
