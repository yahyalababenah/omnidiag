"""
OmniDiag Config Loader
======================
Centralized YAML config loader for all OmniDiag scripts.
Provides a single `load_config(disease_name)` function that returns
a dictionary of paths and parameters from the disease's YAML config.

Usage:
    from configs.config_loader import load_config
    cfg = load_config("heart_disease")
    data_path = cfg["data"]["processed_path"] + cfg["data"]["heuristic_file"]
"""

import os
import yaml
from typing import Dict, Any

# Project root is always one level up from configs/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


def load_config(disease_name: str) -> Dict[str, Any]:
    """
    Load a disease configuration from its YAML file.

    Args:
        disease_name: The disease identifier (e.g., "heart_disease").
                      Must match the filename: configs/{disease_name}.yaml

    Returns:
        A dictionary containing the full configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = os.path.join(CONFIGS_DIR, f"{disease_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Ensure configs/{disease_name}.yaml exists."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(config: Dict[str, Any], *keys: str) -> str:
    """
    Resolve a path from the config relative to PROJECT_ROOT.

    Args:
        config: The loaded configuration dictionary.
        *keys: Sequence of keys to traverse (e.g., "data", "processed_path").

    Returns:
        An absolute path string.

    Example:
        >>> cfg = load_config("heart_disease")
        >>> path = resolve_path(cfg, "data", "processed_path")
        '/workspaces/Heart_Disease_Project/data/heart_disease/processed/'
    """
    path = config
    for key in keys:
        path = path[key]
    return os.path.join(PROJECT_ROOT, path)


def resolve_file_path(config: Dict[str, Any], dir_keys: list, filename_key: str) -> str:
    """
    Resolve a full file path from config directory keys + filename key.

    Args:
        config: The loaded configuration dictionary.
        dir_keys: List of keys to traverse for the directory (e.g., ["data", "processed_path"]).
        filename_key: The key for the filename within the config.

    Returns:
        An absolute file path string.
    """
    directory = resolve_path(config, *dir_keys)
    filename = config
    for key in dir_keys[:-1]:
        filename = filename[key]
    filename = filename[filename_key]
    return os.path.join(directory, filename)
