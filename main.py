#!/usr/bin/env python3
"""
OckBench - LLM Benchmarking Tool for Reasoning Tasks

Main CLI entry point for running benchmarks using Hydra.
"""
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from src.core.runner import run_benchmark

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> int:
    """
    Main entry point for running benchmarks.
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        # Resolve config and convert to container
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        
        # Note: Provider and task configs are merged into the global namespace via '# @package _global_'
        # so no manual flattening is required.
        
        # Resolve paths that are relative to project root
        # Hydra changes the working directory, so we need to use absolute paths
        if 'dataset_path' in cfg_dict and cfg_dict['dataset_path']:
            cfg_dict['dataset_path'] = hydra.utils.to_absolute_path(cfg_dict['dataset_path'])
        
        # Extract special args that are passed as arguments to run_benchmark
        output_dir = cfg_dict.pop('output_dir', 'results')
        log_dir = cfg_dict.pop('log_dir', None)
        
        # Remove Hydra-specific keys if any (handled by OmegaConf.to_container usually, but good to be safe)
        # Note: cfg_dict will be passed as **config_overrides to load_config
        # load_config will override the empty config_path defaults with these values.
        
        # Run benchmark
        experiment = run_benchmark(
            config_path=None,  # We are providing full config via overrides
            output_dir=output_dir,
            log_dir=log_dir,
            **cfg_dict
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Dataset: {experiment.dataset_name}")
        print(f"Model: {experiment.config.model}")
        print(f"Accuracy: {experiment.summary.accuracy:.2f}% "
              f"({experiment.summary.correct_count}/{experiment.summary.total_problems})")
        print(f"Total Tokens: {experiment.summary.total_tokens:,}")
        print(f"  Prompt: {experiment.summary.total_prompt_tokens:,}")
        print(f"  Answer: {experiment.summary.total_answer_tokens:,}")
        print(f"  Reasoning: {experiment.summary.total_reasoning_tokens:,}")
        print(f"  Output: {experiment.summary.total_output_tokens:,}")
        print(f"Avg Tokens/Problem: {experiment.summary.avg_tokens_per_problem:.1f}")
        print(f"Duration: {experiment.summary.total_duration:.2f}s")
        print("=" * 80)
        
        # Exit with error code if there were errors
        if experiment.summary.error_count > 0:
            print(f"\nWarning: {experiment.summary.error_count} problems had errors")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    main()
