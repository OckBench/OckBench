"""
Configuration loader for OckBench.
"""
import os
import yaml
from typing import Optional, Dict, Any
from pathlib import Path

from .schemas import BenchmarkConfig


def load_config(config_path: Optional[str] = None, **overrides) -> BenchmarkConfig:
    """
    Load configuration from YAML file and apply overrides.
    
    Args:
        config_path: Path to YAML config file (optional)
        **overrides: Additional config parameters to override file settings
    
    Returns:
        BenchmarkConfig: Validated configuration object
    """
    config_dict = {}
    
    # Load from YAML file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
    
    # Apply environment variables for sensitive data
    config_dict = _apply_env_vars(config_dict)
    
    # Apply overrides from function arguments
    config_dict.update({k: v for k, v in overrides.items() if v is not None})
    
    # Validate and create BenchmarkConfig
    config = BenchmarkConfig(**config_dict)
    
    return config


def _apply_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variables to config.
    
    Environment variable priority:
    1. If api_key is not in config, try provider-specific env vars
    2. If base_url is not in config for generic provider, try env var
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        Updated configuration dictionary
    """
    # Handle API key from environment
    if not config_dict.get('api_key'):
        provider = config_dict.get('provider', '').lower()
        
        # Try provider-specific environment variables
        env_key_map = {
            'openai': 'OPENAI_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'generic': 'API_KEY'
        }
        
        env_var = env_key_map.get(provider)
        if env_var and os.getenv(env_var):
            config_dict['api_key'] = os.getenv(env_var)
    
    # Handle base URL from environment for generic provider
    if config_dict.get('provider') == 'generic' and not config_dict.get('base_url'):
        if os.getenv('API_BASE_URL'):
            config_dict['base_url'] = os.getenv('API_BASE_URL')
    
    return config_dict


def save_config(config: BenchmarkConfig, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: BenchmarkConfig object
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and remove sensitive data
    config_dict = config.model_dump()
    
    # Optionally mask API key
    if config_dict.get('api_key'):
        config_dict['api_key'] = '***MASKED***'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

