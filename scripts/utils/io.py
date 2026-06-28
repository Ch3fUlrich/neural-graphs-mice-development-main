"""Compatibility wrapper for the project YAML config loader.

Prefer importing from :mod:`ngmd.config` in new code. This module remains so
existing references to ``scripts/io.py`` keep working.
"""

from ngmd.config import get_config_section, load_config

__all__ = ["get_config_section", "load_config"]

