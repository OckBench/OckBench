"""
Code evaluation module with code extraction, execution, and test validation.

Safely executes Python code in subprocess with timeout protection.
"""
import re
import logging
import subprocess
import tempfile
import os
from typing import Tuple, Optional, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeEvaluator:
    """
    Evaluator for code generation problems.
    
    Extracts code from model responses using multiple patterns,
    executes in subprocess with timeout, and validates against test cases.
    """
    
    def __init__(self, timeout: int = 5):
        """
        Initialize code evaluator.
        
        Args:
            timeout: Maximum execution time in seconds (default: 5)
        """
        self.timeout = timeout
        self.extraction_patterns = [
            # Pattern 0: Solution tags (explicit format requested in prompt)
            (r'<solution>\s*(.*?)\s*</solution>', 'solution_tags', re.DOTALL),
            
            # Pattern 1: Python code block with language hint
            (r'```python\s*\n(.*?)\n```', 'markdown_python', re.DOTALL),
            
            # Pattern 2: Generic code block (no language)
            (r'```\s*\n(.*?)\n```', 'markdown_generic', re.DOTALL),
            
            # Pattern 3: Function definition to end (greedy)
            (r'(def\s+\w+\s*\([^)]*\):[^\n]*(?:\n(?:    |\t).*)*)', 'function_def', re.MULTILINE),
            
            # Pattern 4: Class definition
            (r'(class\s+\w+.*?:\s*\n(?:(?:    |\t).*\n)*)', 'class_def', re.MULTILINE),
        ]
    
    def extract_code(self, response: str) -> Tuple[Optional[str], str]:
        """
        Extract Python code from model response using multiple patterns.
        
        Args:
            response: Model response text
        
        Returns:
            Tuple of (extracted_code, extraction_method)
        """
        if not response or not response.strip():
            return None, 'empty_response'
        
        # Try each pattern in order of specificity
        for pattern, method_name, flags in self.extraction_patterns:
            matches = re.findall(pattern, response, flags)
            
            if matches:
                # Take the last match (usually the final/complete version)
                code = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
                code = code.strip()
                
                if code:
                    logger.debug(f"Extracted code using method '{method_name}'")
                    return code, method_name
        
        # Fallback: Try to extract any Python-like code
        # Look for function definitions anywhere in the response
        func_pattern = r'def\s+\w+\s*\([^)]*\):'
        if re.search(func_pattern, response):
            # Extract from first def to end of indented block
            lines = response.split('\n')
            code_lines = []
            in_function = False
            
            for line in lines:
                if re.match(r'^\s*def\s+\w+', line):
                    in_function = True
                    code_lines.append(line)
                elif in_function:
                    if line.strip() and not line.startswith((' ', '\t')):
                        # End of indented block
                        break
                    code_lines.append(line)
            
            if code_lines:
                code = '\n'.join(code_lines).strip()
                logger.debug(f"Extracted code using fallback method")
                return code, 'fallback_extraction'
        
        # No code found
        logger.warning(f"Could not extract code from response: {response[:100]}...")
        return None, 'no_match'
    
    def _create_test_script(self, code: str, test_cases: List[str]) -> str:
        """
        Create a complete Python script with code and test cases.
        
        Args:
            code: Extracted code to test
            test_cases: List of test assertions
        
        Returns:
            str: Complete Python script
        """
        script_parts = [
            "# Extracted code",
            code,
            "",
            "# Test cases",
            "def run_tests():",
            "    passed = 0",
            "    failed = 0",
            "    total = 0",
            ""
        ]
        
        for i, test in enumerate(test_cases):
            # Escape the test string for safe inclusion in f-string
            # Replace single quotes with double quotes to avoid syntax errors
            safe_test = test.replace("'", '"')
            script_parts.extend([
                f"    # Test {i + 1}",
                f"    total += 1",
                f"    try:",
                f"        {test}",
                f"        passed += 1",
                f"    except AssertionError as e:",
                f"        failed += 1",
                f"        print(f'FAILED Test {i + 1}: {safe_test}')",
                f"    except Exception as e:",
                f"        failed += 1",
                f"        print(f'ERROR Test {i + 1}: {{type(e).__name__}}: {{e}}')",
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
    
    def execute_code(
        self,
        code: str,
        test_cases: List[str]
    ) -> Tuple[bool, int, int, Optional[str]]:
        """
        Execute code with test cases in subprocess with timeout.
        
        Args:
            code: Python code to execute
            test_cases: List of test assertions
        
        Returns:
            Tuple of (all_passed, tests_passed, tests_total, error_message)
        """
        if not code:
            return False, 0, len(test_cases), "No code to execute"
        
        if not test_cases:
            return False, 0, 0, "No test cases provided"
        
        # Create test script
        script = self._create_test_script(code, test_cases)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(script)
            temp_file = f.name
        
        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode
            
            # Parse results from output
            tests_passed = 0
            tests_total = len(test_cases)
            error_message = None
            
            # Look for RESULTS line
            results_pattern = r'RESULTS: (\d+)/(\d+) tests passed'
            match = re.search(results_pattern, stdout)
            
            if match:
                tests_passed = int(match.group(1))
                tests_total = int(match.group(2))
            
            # Check for errors
            if returncode != 0:
                # Collect error information
                error_lines = []
                
                # Get failed test info from stdout
                for line in stdout.split('\n'):
                    if line.startswith('FAILED') or line.startswith('ERROR'):
                        error_lines.append(line)
                
                # Get syntax/runtime errors from stderr
                if stderr:
                    # Extract relevant error info (last few lines)
                    stderr_lines = stderr.strip().split('\n')
                    relevant_stderr = stderr_lines[-3:] if len(stderr_lines) > 3 else stderr_lines
                    error_lines.extend(relevant_stderr)
                
                error_message = '\n'.join(error_lines) if error_lines else "Execution failed"
            
            all_passed = (tests_passed == tests_total and returncode == 0)
            
            logger.debug(
                f"Execution complete: {tests_passed}/{tests_total} passed, "
                f"returncode={returncode}"
            )
            
            return all_passed, tests_passed, tests_total, error_message
        
        except subprocess.TimeoutExpired:
            error_message = f"Execution timeout after {self.timeout} seconds"
            logger.warning(error_message)
            return False, 0, len(test_cases), error_message
        
        except Exception as e:
            error_message = f"Execution error: {type(e).__name__}: {str(e)}"
            logger.error(error_message)
            return False, 0, len(test_cases), error_message
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")
    
    def evaluate(self, response: str, test_cases: List[str]):
        """Evaluate model response against test cases."""
        from . import EvalResult
        extracted_code, extraction_method = self.extract_code(response)

        if not extracted_code:
            return EvalResult(
                is_correct=False, extracted_answer=None, extraction_method=extraction_method,
                tests_passed=0, tests_total=len(test_cases), execution_error="Failed to extract code"
            )

        all_passed, tests_passed, tests_total, error_message = self.execute_code(extracted_code, test_cases)
        return EvalResult(
            is_correct=all_passed, extracted_answer=extracted_code, extraction_method=extraction_method,
            tests_passed=tests_passed, tests_total=tests_total, execution_error=error_message
        )
