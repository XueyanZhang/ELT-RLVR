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
SOURCE_DATA_DIR = REPO_ROOT / "data/source"

FETCH_COMMAND = "python scripts/fetch_source_data.py --max-rows 10000"

if str(OFFICIAL_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_EVAL_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_benchmark_data: requires the benchmark data fetched under data/",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the data-dependent tests cleanly when the data has not been fetched.

    `data/` is deliberately not in the repository - it is several gigabytes and
    is pulled on demand. Without this hook a fresh clone reports a pile of
    FileNotFoundError instead of naming the command to run, which is the first
    thing anyone checking this project out would hit.
    """
    if SOURCE_DATA_DIR.is_dir():
        return
    skip = pytest.mark.skip(reason=f"benchmark data not fetched - run: {FETCH_COMMAND}")
    for item in items:
        if item.get_closest_marker("needs_benchmark_data"):
            item.add_marker(skip)


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
