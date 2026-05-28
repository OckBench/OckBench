"""Data loaders for different dataset formats."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..core.schemas import Problem


class DataLoader(ABC):
    @abstractmethod
    def load(self) -> List[Problem]:
        pass


class JSONLDataLoader(DataLoader):
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")

    def load(self) -> List[Problem]:
        problems = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    problems.append(Problem(**data))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    raise ValueError(f"Error parsing problem at line {line_num}: {e}")
        return problems


class MBPPDataLoader(DataLoader):
    """Loader for MBPP format with test cases in metadata."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")

    def load(self) -> List[Problem]:
        problems = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    if 'problem' in data and 'answer' in data:
                        problem_text = data.get('problem', '')
                        answer = data.get('answer', '')
                        problem_id = data.get('id', line_num)
                        metadata = data.get('metadata', {})

                        if 'test_cases' not in metadata:
                            test_list = metadata.get('test_list', [])
                            challenge_test_list = metadata.get('challenge_test_list', [])
                            metadata['test_cases'] = test_list + challenge_test_list

                        enhanced_text = problem_text
                        test_list = metadata.get('test_list', [])
                        if test_list:
                            enhanced_text += "\n\nYour code should pass these tests:\n"
                            for test in test_list[:3]:
                                enhanced_text += f"  {test}\n"

                        problems.append(Problem(
                            problem=enhanced_text, answer=answer,
                            id=problem_id, metadata=metadata,
                        ))
                    else:
                        # Old nested format (backward compatibility)
                        if 'doc' in data:
                            doc = data['doc']
                            task_id = doc.get('task_id', data.get('doc_id', line_num))
                            text = doc.get('text', '')
                            code = doc.get('code', '')
                            test_list = doc.get('test_list', [])
                            challenge_test_list = doc.get('challenge_test_list', [])
                        else:
                            task_id = data.get('task_id', data.get('id', line_num))
                            text = data.get('text', data.get('problem', ''))
                            code = data.get('code', '')
                            test_list = data.get('test_list', [])
                            challenge_test_list = data.get('challenge_test_list', [])

                        all_tests = test_list + challenge_test_list
                        enhanced_text = text
                        if test_list:
                            enhanced_text += "\n\nYour code should pass these tests:\n"
                            for test in test_list[:3]:
                                enhanced_text += f"  {test}\n"

                        problems.append(Problem(
                            problem=enhanced_text, answer=code, id=task_id,
                            metadata={
                                'test_cases': all_tests,
                                'test_list': test_list,
                                'challenge_test_list': challenge_test_list,
                                'reference_code': code,
                                'original_text': text,
                            },
                        ))

                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    raise ValueError(f"Error parsing problem at line {line_num}: {e}")
        return problems


def get_loader(filepath: str, **kwargs) -> DataLoader:
    """Get appropriate data loader for filepath."""
    if not filepath:
        raise ValueError("filepath is required")
    if 'mbpp' in filepath.lower():
        return MBPPDataLoader(filepath)
    return JSONLDataLoader(filepath)
