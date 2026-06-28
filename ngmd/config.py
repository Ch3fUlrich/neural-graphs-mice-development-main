"""Project configuration loaded from ``configs/config.yaml``."""

import os
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


def _default_config_path():
    candidates = [DEFAULT_CONFIG_PATH]
    cwd = Path.cwd().resolve()
    candidates.extend(
        parent / "configs" / "config.yaml" for parent in (cwd, *cwd.parents)
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return DEFAULT_CONFIG_PATH


def load_config(config_path=None):
    """Load the project YAML configuration.

    Parameters
    ----------
    config_path : str or pathlib.Path, optional
        Path to a YAML config file. When omitted, ``configs/config.yaml`` at the
        repository root is used.

    Returns
    -------
    dict
        Parsed configuration values.
    """
    config_path = (
        Path(config_path) if config_path is not None else _default_config_path()
    )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    paths_config = config.setdefault("paths", {})
    if "NGMD_DATA" in os.environ:
        paths_config["data_dir"] = os.environ["NGMD_DATA"]
    if "NGMD_MODELS" in os.environ:
        paths_config["models_dir"] = os.environ["NGMD_MODELS"]

    return config


def get_config_section(section_name, defaults=None, config_path=None):
    """Load one config section, overlaid on optional defaults."""
    section = dict(defaults or {})
    config = load_config(config_path)
    section.update(config.get(section_name, {}) or {})
    return section
