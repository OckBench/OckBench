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


class HuggingFaceDataLoader(DataLoader):
    """
    Placeholder for future HuggingFace dataset loader.
    
    To implement:
    1. Install datasets library: pip install datasets
    2. Load dataset: dataset = load_dataset(dataset_name, split=split)
    3. Map to Problem objects with appropriate field mapping
    """
    
    def __init__(self, dataset_name: str, split: str = "test", field_mapping: dict = None):
        """
        Initialize HuggingFace dataset loader.
        
        Args:
            dataset_name: Name of dataset on HuggingFace Hub
            split: Dataset split to load (train/test/validation)
            field_mapping: Mapping from HF fields to Problem fields
        """
        self.dataset_name = dataset_name
        self.split = split
        self.field_mapping = field_mapping or {
            'problem': 'question',
            'answer': 'answer',
            'id': 'id'
        }
        raise NotImplementedError(
            "HuggingFaceDataLoader is not yet implemented. "
            "Use JSONLDataLoader for local files."
        )
    
    def load(self) -> List[Problem]:
        """Load from HuggingFace dataset (not implemented)."""
        raise NotImplementedError()


class MBPPDataLoader(DataLoader):
    """
    Loader for MBPP (Mostly Basic Python Problems) format datasets.
    
    Expected format: Each line is a JSON object with fields:
    - doc_id: int (problem identifier)
    - doc: dict with:
        - task_id: int
        - text: str (problem description)
        - code: str (reference solution)
        - test_list: list[str] (basic test assertions)
        - challenge_test_list: list[str] (additional test assertions)
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
                    
                    # Handle MBPP format with nested 'doc' field
                    if 'doc' in data:
                        doc = data['doc']
                        task_id = doc.get('task_id', data.get('doc_id', line_num))
                        text = doc.get('text', '')
                        code = doc.get('code', '')
                        test_list = doc.get('test_list', [])
                        challenge_test_list = doc.get('challenge_test_list', [])
                    else:
                        # Handle flat format
                        task_id = data.get('task_id', data.get('id', line_num))
                        text = data.get('text', data.get('problem', ''))
                        code = data.get('code', '')
                        test_list = data.get('test_list', [])
                        challenge_test_list = data.get('challenge_test_list', [])
                    
                    # Combine all test cases
                    all_tests = test_list + challenge_test_list
                    
                    # Enhance problem text with test cases to show expected function signature
                    enhanced_text = text
                    if test_list:
                        enhanced_text += "\n\nYour code should pass these tests:\n"
                        for test in test_list[:3]:  # Show first 3 tests
                            enhanced_text += f"  {test}\n"
                    
                    # Create Problem object
                    problem = Problem(
                        problem=enhanced_text,
                        answer=code,  # Reference solution (not used for evaluation)
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


def get_loader(filepath: str = None, dataset_name: str = None, **kwargs) -> DataLoader:
    """
    Factory function to get appropriate data loader.
    
    Args:
        filepath: Path to local file (for JSONL loader)
        dataset_name: HuggingFace dataset name (for HF loader)
        **kwargs: Additional arguments for loader
    
    Returns:
        DataLoader: Appropriate data loader instance
    """
    if filepath:
        filepath_obj = Path(filepath)
        
        # Auto-detect MBPP format
        if 'mbpp' in filepath.lower():
            return MBPPDataLoader(filepath)
        else:
            return JSONLDataLoader(filepath)
    elif dataset_name:
        return HuggingFaceDataLoader(dataset_name, **kwargs)
    else:
        raise ValueError("Must provide either filepath or dataset_name")

