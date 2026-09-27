"""Access to the pinned ELT-Bench evaluation code.

The benchmark's comparator is the definition of correctness we are scoring
against, so we import it rather than copy it. It lives in a git submodule
pinned to one commit, which also lets a reviewer confirm with `git diff` that
we did not weaken the grader.
"""
from __future__ import annotations

import sys
from functools import cache
from pathlib import Path
from types import ModuleType

EVAL_DIR = Path(__file__).resolve().parents[1] / "third_party/ELT-Bench/evaluation"


@cache
def official_stage2() -> ModuleType:
    """Import upstream `eva_stage2`, adding the submodule to `sys.path` once."""
    if not EVAL_DIR.is_dir():
        raise RuntimeError(
            f"ELT-Bench submodule missing at {EVAL_DIR}. "
            "Run: git submodule update --init --recursive"
        )
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import eva_stage2

    return eva_stage2
