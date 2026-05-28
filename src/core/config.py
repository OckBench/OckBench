"""Configuration loader for OckBench."""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

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

    config_dict = _apply_env_vars(config_dict)
    config_dict.update({k: v for k, v in overrides.items() if v is not None})

    return BenchmarkConfig(**config_dict)


def _apply_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variables for API keys (non-chat_completion providers only)."""
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

    return config_dict


def save_config(config: BenchmarkConfig, output_path: str):
    """Save configuration to YAML file (API key masked)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()
    if config_dict.get('api_key'):
        config_dict['api_key'] = '***MASKED***'

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
