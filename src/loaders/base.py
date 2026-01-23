"""
Data loaders for different dataset formats.
"""
import json
from abc import ABC, abstractmethod
from typing import List
from pathlib import Path

from ..core.schemas import Problem


class DataLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    This design allows easy extension to support different data sources
    (e.g., HuggingFace datasets, CSV files, etc.)
    """
    
    @abstractmethod
    def load(self) -> List[Problem]:
        """
        Load dataset and return list of Problem objects.
        
        Returns:
            List[Problem]: List of problems from the dataset
        """
        pass


class JSONLDataLoader(DataLoader):
    """
    Loader for JSONL (JSON Lines) format datasets.
    
    Expected format: Each line is a JSON object with fields:
    - problem: str (question text)
    - answer: any (ground truth answer)
    - id: any (problem identifier)
    """
    
    def __init__(self, filepath: str):
        """
        Initialize JSONL data loader.
        
        Args:
            filepath: Path to JSONL file
        """
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")
    
    def load(self) -> List[Problem]:
        """
        Load problems from JSONL file.
        
        Returns:
            List[Problem]: List of Problem objects
        """
        problems = []
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    data = json.loads(line)
                    problem = Problem(**data)
                    problems.append(problem)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    raise ValueError(f"Error parsing problem at line {line_num}: {e}")
        
        return problems


class MBPPDataLoader(DataLoader):
    """
    Loader for MBPP (Mostly Basic Python Problems) format datasets.
    
    Now uses standard format with metadata containing test cases.
    Expected format: Each line is a JSON object with fields:
    - problem: str (problem description)
    - answer: str (reference solution code)
    - id: int (problem identifier)
    - metadata: dict with:
        - test_cases: list[str] (all test assertions)
        - test_list: list[str] (basic test assertions)
        - challenge_test_list: list[str] (additional test assertions)
        - reference_code: str (reference solution)
    """
    
    def __init__(self, filepath: str):
        """
        Initialize MBPP data loader.
        
        Args:
            filepath: Path to MBPP JSONL file
        """
        self.filepath = Path(filepath)
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")
    
    def load(self) -> List[Problem]:
        """
        Load problems from MBPP JSONL file.
        
        Returns:
            List[Problem]: List of Problem objects with test cases in metadata
        """
        problems = []
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Handle standard format (new format)
                    if 'problem' in data and 'answer' in data:
                        # Standard format - use directly
                        problem_text = data.get('problem', '')
                        answer = data.get('answer', '')
                        problem_id = data.get('id', line_num)
                        metadata = data.get('metadata', {})
                        
                        # Ensure test_cases are in metadata
                        if 'test_cases' not in metadata:
                            test_list = metadata.get('test_list', [])
                            challenge_test_list = metadata.get('challenge_test_list', [])
                            metadata['test_cases'] = test_list + challenge_test_list
                        
                        # Enhance problem text with test cases to show expected function signature
                        enhanced_text = problem_text
                        test_list = metadata.get('test_list', [])
                        if test_list:
                            enhanced_text += "\n\nYour code should pass these tests:\n"
                            for test in test_list[:3]:  # Show first 3 tests
                                enhanced_text += f"  {test}\n"
                        
                        problem = Problem(
                            problem=enhanced_text,
                            answer=answer,
                            id=problem_id,
                            metadata=metadata
                        )
                        problems.append(problem)
                    else:
                        # Handle old nested format (backward compatibility)
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
                        
                        problem = Problem(
                            problem=enhanced_text,
                            answer=code,
                            id=task_id,
                            metadata={
                                'test_cases': all_tests,
                                'test_list': test_list,
                                'challenge_test_list': challenge_test_list,
                                'reference_code': code,
                                'original_text': text
                            }
                        )
                        problems.append(problem)
                    
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

