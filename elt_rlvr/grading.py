"""Grading a finished rollout.

The reward is a property of the warehouse the rollout left behind, not of what
the agent said about it. Grading reads the tables that are actually there and
compares them against the benchmark's ground truth; a model that was never
built, or built from SQL that failed, is simply absent and scores zero.

One pass produces both verdicts: the graded reward that trains the policy, and
the benchmark's own pass/fail, reported unchanged so the two remain comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import pandas as pd

from elt_rlvr.reward import ModelScore, score_model, weighted_baseline
from elt_rlvr.task import TaskSpec

#: Weights for a rollout that performs the load as well as the transform.
EL_WEIGHT = 0.3
TRANSFORM_WEIGHT = 0.7


@dataclass(frozen=True)
class RolloutGrade:
    """What a rollout earned, and everything needed to explain it."""

    reward: float
    r_transform: float
    r_el: float
    official_srdt: float
    official_srdel: bool
    models: tuple[ModelScore, ...]
    metrics: dict[str, float]


@lru_cache(maxsize=None)
def _baselines_for(name: str, ground_truth_dir: str, targets: tuple[str, ...],
                   keys: tuple[tuple[str, tuple[str, ...]], ...]) -> dict[str, float]:
    key_map = dict(keys)
    baselines = {}
    for model in targets:
        gt = _read_ground_truth(f"{ground_truth_dir}/{model}.csv")
        baselines[model] = weighted_baseline(gt, key_map.get(model, ()))
    return baselines


def model_baselines(task: TaskSpec) -> dict[str, float]:
    """What guessing scores on each of a task's models.

    Depends only on the ground truth, so it is computed once per task and
    reused by every rollout - the alternative would be re-deriving the same
    number for all eight members of a group, on every step.
    """
    return dict(
        _baselines_for(
            task.name,
            str(task.ground_truth_dir),
            tuple(task.targets),
            tuple((m, tuple(k)) for m, k in task.keys.items()),
        )
    )


@lru_cache(maxsize=256)
def _read_ground_truth(path: str) -> pd.DataFrame:
    """Ground truth is immutable, so each file is read once per process."""
    return pd.read_csv(path, dtype=str, keep_default_na=True, low_memory=False)


def _fetch(warehouse: Any, schema: str, table: str) -> pd.DataFrame | None:
    try:
        return warehouse.fetch_table("elt", schema, table).astype(str)
    except Exception:
        return None  # never built, or built by SQL that failed


def grade_rollout(
    task: TaskSpec,
    warehouse: Any,
    *,
    baselines: Mapping[str, float] | None = None,
    include_load: bool = False,
) -> RolloutGrade:
    """Score the warehouse a rollout left behind.

    `include_load` is for the stage where the agent performs the extract-load
    itself; with the tables loaded for it there is nothing to grade there, and
    folding in a term that is always 1.0 would only shift every reward by a
    constant, which group-centred advantages discard anyway.
    """
    if baselines is None:
        baselines = model_baselines(task)

    scores = tuple(
        score_model(
            model,
            _read_ground_truth(str(task.ground_truth_dir / f"{model}.csv")),
            _fetch(warehouse, task.name, model),
            task.keys.get(model, ()),
            baselines[model],
        )
        for model in task.targets
    )

    r_transform = sum(s.reward for s in scores) / len(scores) if scores else 0.0
    official_srdt = (
        sum(1.0 for s in scores if s.official_pass) / len(scores) if scores else 0.0
    )

    r_el, official_srdel = _grade_load(task, warehouse) if include_load else (1.0, True)
    reward = (
        EL_WEIGHT * r_el + TRANSFORM_WEIGHT * r_transform if include_load else r_transform
    )

    metrics: dict[str, float] = {
        "reward": reward,
        "r_transform": r_transform,
        "official_srdt": official_srdt,
        "official_srdel": float(official_srdel),
        "models_attempted": float(sum(1 for s in scores if s.error is None)),
    }
    if include_load:
        metrics["r_el"] = r_el
    for score in scores:
        metrics[f"reward/{score.name}"] = score.reward
        metrics[f"rows/{score.name}"] = float(score.n_pred_rows)

    return RolloutGrade(
        reward, r_transform, r_el, official_srdt, official_srdel, scores, metrics
    )


def _grade_load(task: TaskSpec, warehouse: Any) -> tuple[float, bool]:
    """Share of source tables present at exactly the expected row count.

    Exactness is what makes this hard to fake: a table can be conjured, but not
    with the right number of rows in it.
    """
    if not task.expected_rows:
        return 1.0, True
    correct = 0
    for table, expected in task.expected_rows.items():
        try:
            correct += int(warehouse.table_size("elt", task.name, table) == expected)
        except Exception:
            pass
    return correct / len(task.expected_rows), correct == len(task.expected_rows)
