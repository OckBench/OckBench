#!/usr/bin/env python3
"""
Convert GPQA-Diamond dataset to OckBench format.

GPQA (Graduate-Level Google-Proof Q&A) is a multiple-choice science benchmark
covering physics, chemistry, biology, and astronomy at graduate level.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def categorize_topic(question: str) -> str:
    """Heuristically categorize question by scientific topic."""
    q_lower = question.lower()

    topics = {
        'physics': ['quantum', 'particle', 'energy', 'force', 'electron', 'photon',
                   'relativity', 'magnetic', 'electric', 'wave', 'momentum', 'velocity',
                   'spin', 'angular momentum', 'hamiltonian', 'lagrangian', 'field theory'],
        'chemistry': ['molecule', 'reaction', 'compound', 'element', 'bond', 'acid',
                     'base', 'organic', 'synthesis', 'catalyst', 'reagent', 'nmr',
                     'spectroscopy', 'stereochemistry', 'oxidation', 'reduction'],
        'biology': ['cell', 'gene', 'protein', 'dna', 'rna', 'enzyme', 'organism',
                   'species', 'evolution', 'mutation', 'chromosome', 'genome', 'amino acid'],
        'astronomy': ['planet', 'star', 'galaxy', 'orbit', 'solar', 'exoplanet',
                     'black hole', 'universe', 'cosmic', 'luminosity', 'redshift'],
    }

    for topic, keywords in topics.items():
        if any(kw in q_lower for kw in keywords):
            return topic
    return 'general_science'


def convert_gpqa_problem(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a GPQA problem to OckBench format."""

    question = example['question']
    answer = example['answer'].strip().upper()  # A, B, C, or D

    # Categorize the topic
    topic = categorize_topic(question)

    # Build the problem with instructions for the model
    problem_text = f"""Answer the following multiple choice question. After your reasoning, clearly state your final answer as a single letter (A, B, C, or D).

{question}

Provide your answer in the format: "The answer is [LETTER]." where [LETTER] is A, B, C, or D."""

    return {
        "problem": problem_text,
        "answer": answer,
        "id": f"GPQA-Diamond-{idx}",
        "metadata": {
            "source": "gpqa_diamond",
            "topic": topic,
            "answer_type": "multiple_choice",
            "choices": ["A", "B", "C", "D"],
            "original_question": question,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Convert GPQA-Diamond to OckBench format")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for JSONL files"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading GPQA-Diamond dataset...")
    ds = load_dataset("fingertap/GPQA-Diamond")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = []
    topic_counts = {}

    for idx, example in enumerate(ds['test']):
        converted = convert_gpqa_problem(example, idx)
        problems.append(converted)

        topic = converted['metadata']['topic']
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    # Save to JSONL
    output_file = output_dir / "GPQA_Diamond.jsonl"
    with open(output_file, 'w') as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(problems)} problems to {output_file}")
    print(f"\nTopic distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
