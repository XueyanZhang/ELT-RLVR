"""Shared test fixtures.

`OFFICIAL_EVAL_DIR` is the pinned ELT-Bench submodule's evaluation package. We
import the official comparator from it directly rather than reimplementing it,
so any drift between our scoring and the benchmark's is caught by tests instead
of going unnoticed.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_EVAL_DIR = REPO_ROOT / "third_party/ELT-Bench/evaluation"
FIXTURE_GT_DIR = Path(__file__).resolve().parent / "fixtures/gt_snowflake"

if str(OFFICIAL_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_EVAL_DIR))


@pytest.fixture(scope="session")
def official():
    """The upstream stage-2 comparator, imported from the pinned submodule."""
    import eva_stage2

    return eva_stage2


@pytest.fixture(scope="session")
def gt_csv_paths() -> list[Path]:
    paths = sorted(FIXTURE_GT_DIR.glob("*/*.csv"))
    assert paths, f"no ground-truth fixtures under {FIXTURE_GT_DIR}"
    return paths


def frame(**columns) -> "pd.DataFrame":
    """Build an all-string frame, matching how both sides are read for scoring."""
    import pandas as pd

    return pd.DataFrame(
        {name: [str(v) for v in values] for name, values in columns.items()}, dtype=str
    )
