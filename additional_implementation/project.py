"""Bindings to the original RQMoE/RecBole project."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNING_ROOT = REPOSITORY_ROOT / "exps" / "running"
for import_root in (REPOSITORY_ROOT, RUNNING_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from models.Switch import DeepFM_MoE  # noqa: E402
from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.utils import init_seed  # noqa: E402


__all__ = [
    "Config",
    "DeepFM_MoE",
    "REPOSITORY_ROOT",
    "create_dataset",
    "data_preparation",
    "init_seed",
]
