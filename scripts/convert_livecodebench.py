#!/usr/bin/env python3
"""
Convert LiveCodeBench code_generation dataset to OckBench format.

LiveCodeBench has two types of problems:
1. LeetCode (functional): Function-based testing with starter code
2. AtCoder/Codeforces (stdin): stdin/stdout-based testing

This script converts both types to OckBench's JSONL format.
"""

import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple


def extract_function_name(starter_code: str) -> Optional[str]:
    """Extract the main function name from LeetCode starter code."""
    # Pattern for class method: def methodName(self, ...)
    match = re.search(r'def\s+(\w+)\s*\(\s*self', starter_code)
    if match:
        return match.group(1)

    # Pattern for standalone function: def funcName(...)
    match = re.search(r'def\s+(\w+)\s*\(', starter_code)
    if match:
        return match.group(1)

    return None


def extract_class_name(starter_code: str) -> Optional[str]:
    """Extract the class name from LeetCode starter code."""
    match = re.search(r'class\s+(\w+)', starter_code)
    if match:
        return match.group(1)
    return None


def parse_functional_input(input_str: str) -> List[str]:
    """Parse functional test input into individual arguments."""
    # LeetCode inputs are typically newline-separated values
    # Each line is a JSON-parseable value
    args = []
    for line in input_str.strip().split('\n'):
        line = line.strip()
        if line:
            args.append(line)
    return args


def convert_json_to_python(value: str) -> str:
    """Convert JSON values to Python syntax."""
    # Convert JSON booleans to Python booleans
    value = re.sub(r'\btrue\b', 'True', value)
    value = re.sub(r'\bfalse\b', 'False', value)
    value = re.sub(r'\bnull\b', 'None', value)
    return value


def create_functional_assert(
    class_name: str,
    func_name: str,
    input_str: str,
    output_str: str
) -> str:
    """Create an assert statement for functional tests."""
    args = parse_functional_input(input_str)
    # Convert JSON values to Python
    args = [convert_json_to_python(arg) for arg in args]
    args_str = ', '.join(args)
    expected = convert_json_to_python(output_str.strip())

    # For class methods, instantiate the class first
    if class_name:
        return f"assert {class_name}().{func_name}({args_str}) == {expected}"
    else:
        return f"assert {func_name}({args_str}) == {expected}"


def create_stdin_test_wrapper(input_str: str, output_str: str, test_idx: int) -> str:
    """
    Create a test wrapper for stdin/stdout-based problems.

    Returns a test that:
    1. Mocks stdin with the input
    2. Captures stdout
    3. Compares with expected output
    """
    # Escape the input/output for use in Python strings
    input_escaped = input_str.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
    output_escaped = output_str.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')

    return f"""
def test_case_{test_idx}():
    import sys
    from io import StringIO
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = StringIO('{input_escaped}')
    sys.stdout = StringIO()
    try:
        main()  # or solve() - the solution function
    finally:
        output = sys.stdout.getvalue()
        sys.stdin, sys.stdout = old_stdin, old_stdout
    expected = '{output_escaped}'
    assert output.strip() == expected.strip(), f"Expected {{expected!r}}, got {{output!r}}"
"""


def convert_leetcode_problem(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a LeetCode functional problem to OckBench format."""

    starter_code = example['starter_code']
    func_name = extract_function_name(starter_code)
    class_name = extract_class_name(starter_code)

    if not func_name:
        return None  # Skip if we can't extract function name

    # Parse test cases
    public_tests = json.loads(example['public_test_cases'])
    private_tests = json.loads(example['private_test_cases'])

    # Create assert statements
    test_list = []
    for test in public_tests:
        try:
            assert_stmt = create_functional_assert(
                class_name, func_name, test['input'], test['output']
            )
            test_list.append(assert_stmt)
        except Exception as e:
            print(f"Warning: Could not create test for {example['question_id']}: {e}")
            continue

    challenge_test_list = []
    for test in private_tests:
        try:
            assert_stmt = create_functional_assert(
                class_name, func_name, test['input'], test['output']
            )
            challenge_test_list.append(assert_stmt)
        except Exception as e:
            continue

    if not test_list:
        return None  # Skip if no valid tests

    # Build the problem description
    problem_text = f"{example['question_content']}\n\n"
    if starter_code:
        problem_text += f"Complete the following code:\n```python\n{starter_code}\n```"

    return {
        "problem": problem_text,
        "answer": starter_code,  # Reference is the starter code template
        "id": f"LiveCodeBench-{example['question_id']}",
        "metadata": {
            "source": "livecodebench",
            "platform": example['platform'],
            "difficulty": example['difficulty'],
            "question_title": example['question_title'],
            "contest_id": example['contest_id'],
            "contest_date": str(example['contest_date']),
            "starter_code": starter_code,
            "function_name": func_name,
            "class_name": class_name,
            "test_list": test_list,
            "challenge_test_list": challenge_test_list,
            "test_cases": test_list + challenge_test_list,
        }
    }


def convert_stdin_problem(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Convert a stdin/stdout problem to OckBench format.

    For stdin problems, we store the raw input/output pairs and
    the code evaluator needs to handle them differently.
    """

    # Parse test cases
    public_tests = json.loads(example['public_test_cases'])
    private_tests = json.loads(example['private_test_cases'])

    # Store raw test cases for stdin evaluation
    test_list = []
    for i, test in enumerate(public_tests):
        test_list.append({
            "input": test['input'],
            "output": test['output'],
            "type": "stdin"
        })

    challenge_test_list = []
    for i, test in enumerate(private_tests):
        challenge_test_list.append({
            "input": test['input'],
            "output": test['output'],
            "type": "stdin"
        })

    if not test_list:
        return None

    # Build the problem description
    problem_text = example['question_content']

    # Add sample input/output
    problem_text += "\n\n### Sample Input/Output\n"
    for i, test in enumerate(public_tests[:3]):
        problem_text += f"\n**Input {i+1}:**\n```\n{test['input'].strip()}\n```\n"
        problem_text += f"**Output {i+1}:**\n```\n{test['output'].strip()}\n```\n"

    return {
        "problem": problem_text,
        "answer": "",  # No reference solution available
        "id": f"LiveCodeBench-{example['question_id']}",
        "metadata": {
            "source": "livecodebench",
            "platform": example['platform'],
            "difficulty": example['difficulty'],
            "question_title": example['question_title'],
            "contest_id": example['contest_id'],
            "contest_date": str(example['contest_date']),
            "test_type": "stdin",
            "test_list": test_list,
            "challenge_test_list": challenge_test_list,
            "test_cases": test_list + challenge_test_list,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Convert LiveCodeBench to OckBench format")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--functional-only",
        action="store_true",
        help="Only convert functional (LeetCode) problems"
    )
    parser.add_argument(
        "--stdin-only",
        action="store_true",
        help="Only convert stdin (AtCoder/Codeforces) problems"
    )
    args = parser.parse_args()

    # Import here to avoid loading at module level
    from datasets import load_dataset

    print("Loading LiveCodeBench dataset...")
    ds = load_dataset("livecodebench/code_generation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate by test type
    functional_problems = []
    stdin_problems = []
    skipped = 0

    for idx, example in enumerate(ds['test']):
        public_tests = json.loads(example['public_test_cases'])
        test_type = public_tests[0].get('testtype', 'stdin') if public_tests else 'stdin'

        if test_type == 'functional' and not args.stdin_only:
            converted = convert_leetcode_problem(example, idx)
            if converted:
                functional_problems.append(converted)
            else:
                skipped += 1
        elif test_type == 'stdin' and not args.functional_only:
            converted = convert_stdin_problem(example, idx)
            if converted:
                stdin_problems.append(converted)
            else:
                skipped += 1

    # Save functional problems (LeetCode style - assert-based)
    if functional_problems:
        output_file = output_dir / "LiveCodeBench_functional.jsonl"
        with open(output_file, 'w') as f:
            for problem in functional_problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
        print(f"Saved {len(functional_problems)} functional problems to {output_file}")

    # Save stdin problems (AtCoder/Codeforces style)
    if stdin_problems:
        output_file = output_dir / "LiveCodeBench_stdin.jsonl"
        with open(output_file, 'w') as f:
            for problem in stdin_problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
        print(f"Saved {len(stdin_problems)} stdin problems to {output_file}")

    # Save combined
    all_problems = functional_problems + stdin_problems
    if all_problems:
        output_file = output_dir / "LiveCodeBench.jsonl"
        with open(output_file, 'w') as f:
            for problem in all_problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')
        print(f"Saved {len(all_problems)} total problems to {output_file}")

    print(f"\nSkipped: {skipped} problems (could not convert)")

    # Print statistics
    print("\n=== Conversion Statistics ===")
    print(f"Functional (LeetCode): {len(functional_problems)}")
    print(f"Stdin (AtCoder/CF):    {len(stdin_problems)}")
    print(f"Total converted:       {len(all_problems)}")

    # Show difficulty breakdown
    if all_problems:
        difficulties = {}
        for p in all_problems:
            d = p['metadata']['difficulty']
            difficulties[d] = difficulties.get(d, 0) + 1
        print(f"\nDifficulty breakdown: {difficulties}")


if __name__ == "__main__":
    main()
