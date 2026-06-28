"""Shared pytest configuration and fixtures."""

from pathlib import Path
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

for path in (ROOT, SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

matplotlib.use("Agg")


@pytest.fixture
def similarity_matrix():
    return np.array(
        [
            [0.0, 0.8, 0.1],
            [0.8, 0.0, 0.6],
            [0.1, 0.6, 0.0],
        ]
    )
