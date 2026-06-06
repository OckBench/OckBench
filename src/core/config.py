"""Configuration loader for OckBench."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.request_overrides import redact_config
from .schemas import BenchmarkConfig


def load_config(config_path: Optional[str] = None, **overrides) -> BenchmarkConfig:
    """Load configuration from YAML file and apply overrides."""
    config_dict = {}

    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}

    config_dict.update({k: v for k, v in overrides.items() if v is not None})
    # Resolve env keys AFTER the YAML+override merge so the final provider/judge
    # (which may come from overrides) is what env resolution sees.
    config_dict = apply_env_keys(config_dict)

    return BenchmarkConfig(**config_dict)


def apply_env_keys(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve API keys from the environment for the already-merged config.

    Fills the provider ``api_key`` (non-chat_completion providers) and the math
    judge's ``api_key`` only when not already set, so explicit CLI/YAML values
    always win. Must run on the FINAL merged config — provider/judge may arrive
    via overrides, not the YAML base.
    """
    if not config_dict.get('api_key'):
        provider = config_dict.get('provider', '').lower()

        env_key_map = {
            'openai-responses': ['OPENAI_API_KEY'],
            'anthropic': ['ANTHROPIC_API_KEY'],
            'gemini': ['GEMINI_API_KEY'],
        }

        for env_var in env_key_map.get(provider, []):
            if os.getenv(env_var):
                config_dict['api_key'] = os.getenv(env_var)
                break

    # Resolve the math judge's API key from the environment when configured
    # without one (JUDGE_API_KEY preferred, then OPENAI_API_KEY).
    judge = config_dict.get('judge')
    if isinstance(judge, dict) and not judge.get('api_key'):
        for env_var in ('JUDGE_API_KEY', 'OPENAI_API_KEY'):
            if os.getenv(env_var):
                judge['api_key'] = os.getenv(env_var)
                break

    return config_dict


def save_config(config: BenchmarkConfig, output_path: str):
    """Save configuration to YAML file (API key masked)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = redact_config(config.model_dump())

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
