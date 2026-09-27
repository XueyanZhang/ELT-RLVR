"""Reading one ELT-Bench task into the form a rollout needs.

Every fact about a task - which data models to build, what each one's unique
key is, how many rows each source table should hold - is read from the
benchmark's own files rather than restated here, so the tasks trained on are
the published ones and a reviewer can check that by reading the same files.

The one decision this module makes is which of those files the agent is shown.
That is what distinguishes a curriculum stage: when the source tables have
already been loaded, connection details describe systems the agent has no
reason to reach, so they are withheld until the stage that asks for the load.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Mapping

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "third_party/ELT-Bench"
TASK_DIR = BENCHMARK / "elt-bench/snowflake"
SCHEMA_DIR = BENCHMARK / "elt-bench/schemas"
SOURCE_DIR = REPO / "data/source"
GROUND_TRUTH_DIR = REPO / "data/gt/gt_snowflake"


@dataclass(frozen=True)
class TaskSpec:
    """One task, as a rollout sees it.

    `inputs` is everything the agent may read. The ground truth is not in it,
    is not loaded into the warehouse it queries, and is reachable through no
    tool - the answer key stays with the grader.
    """

    name: str
    targets: tuple[str, ...]
    keys: Mapping[str, tuple[str, ...]]
    expected_rows: Mapping[str, int]
    source_dir: Path
    ground_truth_dir: Path
    inputs: Mapping[str, str]


@cache
def _sort_keys() -> dict:
    return json.loads((BENCHMARK / "evaluation/sort_key.json").read_text())


@cache
def _table_counts() -> dict:
    return json.loads((BENCHMARK / "evaluation/table.json").read_text())


def load_task(name: str, *, include_connections: bool = False) -> TaskSpec:
    """Assemble one task from the benchmark's files and the fetched data.

    Set `include_connections` for the stage where the agent performs the load
    itself; it adds `config.yaml`, which names the source systems and the
    warehouse.
    """
    import yaml

    task_dir = TASK_DIR / name
    if not task_dir.is_dir():
        raise KeyError(f"{name!r} is not an ELT-Bench task ({TASK_DIR} has no such directory)")

    data_model_text = (task_dir / "data_model.yaml").read_text()
    targets = tuple(m["name"] for m in yaml.safe_load(data_model_text)["models"])

    inputs: dict[str, str] = {"data_model.yaml": data_model_text}
    for schema in sorted((SCHEMA_DIR / name).glob("*.csv")):
        inputs[f"schemas/{schema.name}"] = schema.read_text()
    if include_connections:
        inputs["config.yaml"] = (task_dir / "config.yaml").read_text()

    return TaskSpec(
        name=name,
        targets=targets,
        keys={m: tuple(k) for m, k in _sort_keys().get(name, {}).items()},
        expected_rows=dict(_table_counts().get(name, {})),
        source_dir=SOURCE_DIR / name,
        ground_truth_dir=GROUND_TRUTH_DIR / name,
        inputs=inputs,
    )


def available_tasks() -> list[str]:
    """Tasks whose source data and ground truth are both present locally.

    The benchmark defines a hundred; only the ones actually fetched can be
    rolled out, so this is what the dataset is drawn from.
    """
    names = []
    for source in sorted(SOURCE_DIR.glob("*")):
        if not source.is_dir() or not (TASK_DIR / source.name).is_dir():
            continue
        models = _sort_keys().get(source.name, {})
        if not models:
            continue
        if all((GROUND_TRUTH_DIR / source.name / f"{m}.csv").is_file() for m in models):
            names.append(source.name)
    return names
