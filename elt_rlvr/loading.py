"""Loading a task's source tables into a warehouse.

This is the load half of ELT performed for the agent instead of by it. The
transform-only curriculum starts from a warehouse in this state, and the same
routine builds the template file each rollout of a task is reset to - so it
runs once per task rather than once per episode.

Everything lands as text. The benchmark's comparator reads both sides as
strings and coerces per column, so letting a warehouse infer types here would
put a second, different set of decisions between the answer and the grade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadReport:
    """What landed, and whether it matches what the benchmark expects."""

    schema: str
    loaded: dict[str, int] = field(default_factory=dict)
    #: table -> (rows loaded, rows expected)
    mismatches: dict[str, tuple[int, int]] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.missing


def load_sources(
    warehouse: Any,
    schema: str,
    source_dir: Path,
    expected: dict[str, int] | None = None,
) -> LoadReport:
    """Load every CSV in `source_dir` into `schema`, one table per file.

    `expected` is the benchmark's own row count per table. It does two jobs.
    It turns a truncated download into an error at load time rather than a low
    reward later that looks like the policy's fault. And it is the list of
    tables a task actually declares, so files the archive happens to carry
    alongside them are left out: `trains` ships a copy of one source table
    under the task's own name, which is also the name of the data model the
    task asks for, and loading it would put a table exactly where the
    submission belongs.

    Passing no list loads every CSV, which is what exploratory use wants.
    """
    warehouse.query(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    declared = {name.lower() for name in expected} if expected else None

    loaded: dict[str, int] = {}
    for csv in sorted(Path(source_dir).glob("*.csv")):
        table = csv.stem
        if declared is not None and table.lower() not in declared:
            continue
        warehouse.query(
            f'CREATE OR REPLACE TABLE "{schema}"."{table}" AS '
            f"SELECT * FROM read_csv('{csv}', all_varchar = true)"
        )
        loaded[table] = warehouse.table_size("elt", schema, table)

    if expected is None:
        return LoadReport(schema, loaded)

    # A file on disk and the benchmark's table list need not agree on case -
    # `seasonstatus.csv` ships for a table the benchmark calls `SeasonStatus`.
    # The warehouse folds case, so this comparison does too.
    by_lower = {name.lower(): rows for name, rows in loaded.items()}
    mismatches: dict[str, tuple[int, int]] = {}
    missing: list[str] = []
    for name, count in expected.items():
        actual = by_lower.get(name.lower())
        if actual is None:
            missing.append(name)
        elif actual != count:
            mismatches[name] = (actual, count)

    return LoadReport(schema, loaded, mismatches, tuple(sorted(missing)))
