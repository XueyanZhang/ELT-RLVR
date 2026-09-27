"""Scoring a submitted ELT pipeline against ground truth.

The benchmark grades a data model as a single pass/fail: every ground-truth
column must match, or the model counts for nothing. That verdict is what we
report, but it cannot train a policy — a group of rollouts that all fail scores
uniformly, and GRPO centres rewards within a group, so a uniform group yields
zero advantage and is discarded. Partial credit is what makes the gradient
exist at all.

So we score individual cells. The comparison rules are the benchmark's own,
including its tolerances; only the granularity differs.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from elt_rlvr._official import official_stage2

# Upstream's tolerances, restated here so the numbers are visible at the point
# of use. `test_tolerances_match_official` keeps them honest.
RTOL = 1e-2
ATOL = 1e-9


def cell_matches(gt_col: pd.Series, pred_col: pd.Series) -> list[bool]:
    """Per-cell verdicts for one aligned column pair.

    The official comparator decides whether to compare numerically or textually
    by looking at the whole column — "1" and "1.0" are equal in a column that
    coerces cleanly to numbers and different in one that does not. We reuse its
    coercion helper so that choice is made identically, then apply the matching
    rule cell by cell.

    Both columns must already be aligned and of equal length; callers align by
    unique key before scoring.
    """
    if len(gt_col) != len(pred_col):
        raise ValueError(
            f"columns must be aligned before scoring: {len(gt_col)} vs {len(pred_col)}"
        )

    coerce = official_stage2()._to_numeric_if_possible
    gt_num = coerce(gt_col.reset_index(drop=True))
    pred_num = coerce(pred_col.reset_index(drop=True))

    if pd.api.types.is_numeric_dtype(gt_num) and pd.api.types.is_numeric_dtype(pred_num):
        a = gt_num.to_numpy(dtype=float)
        b = pred_num.to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=False)
        return [bool(x) for x in (both_nan | close)]

    return [_values_match(a, b) for a, b in zip(gt_col, pred_col)]


def _values_match(a: object, b: object) -> bool:
    """Textual comparison rule, mirroring the official non-numeric branch."""
    a_null, b_null = pd.isna(a), pd.isna(b)
    if a_null and b_null:
        return True
    if a_null or b_null:
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return bool(a == b)


# --------------------------------------------------------------------------
# Key-based row alignment
# --------------------------------------------------------------------------

# Most of the benchmark's unique keys are plain column names, but 15 of the 252
# are written as SQL fragments. Both shapes that actually occur are recognised;
# anything else is reported rather than guessed at, because the alternative —
# falling back to row order — would make a score depend on how a query happened
# to sort its output.
_CAST_KEY = re.compile(
    r"^\s*cast\s*\(\s*(?P<col>[A-Za-z_]\w*)\s+as\s+\w+\s*\)\s*$", re.IGNORECASE
)
_NULLS_ORDERED_KEY = re.compile(
    r"^\s*(?P<col>[A-Za-z_]\w*)\s+nulls\s+(?:first|last)\s*$", re.IGNORECASE
)

_MISSING_KEY_VALUE = "\x00<null>"


def resolve_key(key: str, columns: Iterable[str]) -> str | None:
    """Map one `sort_key.json` entry onto an actual column name.

    Returns the column as spelled in the frame, or None if the key cannot be
    tied to a column — callers must treat None as an error, never as "ignore
    this key".
    """
    by_lower = {str(c).lower(): str(c) for c in columns}

    candidates = [key]
    for pattern in (_CAST_KEY, _NULLS_ORDERED_KEY):
        match = pattern.match(key)
        if match:
            candidates.append(match.group("col"))

    for candidate in candidates:
        resolved = by_lower.get(candidate.strip().lower())
        if resolved is not None:
            return resolved
    return None


@dataclass(frozen=True)
class Alignment:
    """Which ground-truth rows were paired with which submitted rows."""

    gt_positions: tuple[int, ...]
    pred_positions: tuple[int, ...]
    n_gt_rows: int
    n_pred_rows: int
    error: str | None = None

    @property
    def n_matched(self) -> int:
        return len(self.gt_positions)

    @property
    def key_recall(self) -> float:
        """Share of ground-truth rows the submission accounted for."""
        return self.n_matched / self.n_gt_rows if self.n_gt_rows else 0.0

    @property
    def key_precision(self) -> float:
        """Share of submitted rows that were useful. Padding lowers this."""
        return self.n_matched / self.n_pred_rows if self.n_pred_rows else 0.0


def _comparable_key_values(gt_col: pd.Series, pred_col: pd.Series) -> tuple[list, list]:
    """Render one key column from each side into directly comparable values.

    A key written `cast(id as integer)` warns that the same identifier may
    arrive as "7" or "7.0" depending on the warehouse, so numeric-looking key
    columns are compared as numbers — matching how the official comparator
    sorts them — and everything else as trimmed, case-folded text.
    """
    coerce = official_stage2()._to_numeric_if_possible
    gt_num = coerce(gt_col.reset_index(drop=True))
    pred_num = coerce(pred_col.reset_index(drop=True))

    if pd.api.types.is_numeric_dtype(gt_num) and pd.api.types.is_numeric_dtype(pred_num):
        render = lambda s: [_MISSING_KEY_VALUE if pd.isna(v) else float(v) for v in s]
        return render(gt_num), render(pred_num)

    render = lambda s: [
        _MISSING_KEY_VALUE if pd.isna(v) else str(v).strip().lower() for v in s
    ]
    return render(gt_col), render(pred_col)


def align_by_key(
    gt: pd.DataFrame, pred: pd.DataFrame, keys: Sequence[str]
) -> Alignment:
    """Pair submitted rows with ground-truth rows by unique key.

    Pairing is one-to-one: a ground-truth row is claimed by at most one
    submitted row, so duplicating output rows cannot inflate recall — it only
    dilutes precision.
    """
    n_gt, n_pred = len(gt), len(pred)
    unaligned = partial(Alignment, (), (), n_gt, n_pred)

    if not keys:
        return unaligned(error="no unique key is defined for this data model")

    columns: list[tuple[str, str]] = []
    for key in keys:
        gt_col = resolve_key(key, gt.columns)
        pred_col = resolve_key(key, pred.columns)
        if gt_col is None:
            return unaligned(error=f"key {key!r} does not resolve to a ground-truth column")
        if pred_col is None:
            return unaligned(error=f"key {key!r} does not resolve to a submitted column")
        columns.append((gt_col, pred_col))

    rendered = [_comparable_key_values(gt[g], pred[p]) for g, p in columns]
    gt_keys = list(zip(*(side[0] for side in rendered))) if rendered else []
    pred_keys = list(zip(*(side[1] for side in rendered))) if rendered else []

    available: dict[tuple, deque[int]] = defaultdict(deque)
    for position, key_tuple in enumerate(pred_keys):
        available[key_tuple].append(position)

    gt_positions: list[int] = []
    pred_positions: list[int] = []
    for position, key_tuple in enumerate(gt_keys):
        candidates = available.get(key_tuple)
        if candidates:
            gt_positions.append(position)
            pred_positions.append(candidates.popleft())

    return Alignment(tuple(gt_positions), tuple(pred_positions), n_gt, n_pred)


# --------------------------------------------------------------------------
# Cell-level scoring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellScore:
    """How much of a data model's expected content was actually produced."""

    recall: float
    precision: float
    f1: float
    n_scored_columns: int
    matched_cells: int
    alignment: Alignment


def scored_columns(gt: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    """Ground-truth columns that earn credit: everything but the key.

    The key is what pairs the rows, so a matching key is a precondition of
    being compared at all — scoring it too would pay for the alignment twice.
    """
    key_columns = {resolve_key(key, gt.columns) for key in keys} - {None}
    return [column for column in gt.columns if column not in key_columns]


def cell_f1(gt: pd.DataFrame, pred: pd.DataFrame, keys: Sequence[str]) -> CellScore:
    """Score a submitted table cell by cell, against the expected cells.

    Both ratios are measured per *expected* cell — `n_rows x ground-truth
    columns` — so omitting columns shifts credit from earned to unearned
    instead of shrinking the target. Recall and precision are combined
    harmonically: an arithmetic mean would pay half marks for emitting one
    perfect row out of a thousand.
    """
    alignment = align_by_key(gt, pred, keys)
    columns = scored_columns(gt, keys)
    n_columns = len(columns)

    if alignment.error is not None or n_columns == 0:
        return CellScore(0.0, 0.0, 0.0, n_columns, 0, alignment)

    matched = 0
    if alignment.n_matched:
        gt_rows = gt.iloc[list(alignment.gt_positions)]
        pred_rows = pred.iloc[list(alignment.pred_positions)]
        pred_by_lower = {str(c).lower(): c for c in pred.columns}
        for column in columns:
            submitted = pred_by_lower.get(str(column).lower())
            if submitted is None:
                continue  # column never produced; its cells stay unearned
            matched += sum(cell_matches(gt_rows[column], pred_rows[submitted]))

    expected_cells = alignment.n_gt_rows * n_columns
    emitted_cells = alignment.n_pred_rows * n_columns
    recall = matched / expected_cells if expected_cells else 0.0
    precision = matched / emitted_cells if emitted_cells else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return CellScore(recall, precision, f1, n_columns, matched, alignment)


def _modal_value(column: pd.Series):
    """The single value that would win the most cells in this column."""
    modes = column.mode(dropna=False)
    return modes.iloc[0] if len(modes) else pd.NA


def constant_filled(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The ground truth with every scored column flattened to its own mode.

    Stands for a policy that produces the right rows and guesses every value.
    Using the mode rather than a plausible constant makes it the best such
    policy, so whatever it scores is the ceiling on guessing.
    """
    guess = gt.copy()
    for column in scored_columns(gt, keys):
        guess[column] = _modal_value(gt[column])
    return guess


def chance_baseline(gt: pd.DataFrame, keys: Sequence[str]) -> float:
    """What a policy that understands nothing can still score on this model.

    The stand-in policy knows the right key set and fills every other column
    with a single constant. Using each column's *most common* value makes this
    the best any constant filler could do, so subtracting it cannot flatter a
    real submission.

    Measured over real ground truth this is far from negligible: the benchmark's
    smaller tables are full of columns that never vary, and scoring them as
    earned would reward nothing but copying the schema.
    """
    if gt.empty or not scored_columns(gt, keys):
        return 1.0  # nothing here can distinguish understanding from guessing
    return cell_f1(gt, constant_filled(gt, keys), keys).f1


# --------------------------------------------------------------------------
# Per-model result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelScore:
    """Both verdicts for one data model: what we train on, and what we report."""

    name: str
    weighted_f1: float
    baseline: float
    reward: float
    official_pass: bool
    key_recall: float
    key_precision: float
    n_gt_rows: int
    n_pred_rows: int
    error: str | None = None


def official_verdict(gt: pd.DataFrame, pred: pd.DataFrame, keys: Sequence[str]) -> bool:
    """The benchmark's own pass/fail, computed by the benchmark's own code.

    Reported next to every reward so our numbers stay comparable to published
    results, and so any divergence is visible rather than assumed away.
    """
    upstream = official_stage2()
    try:
        return bool(
            upstream.check_corretness(
                upstream.sort_by_keys(gt, list(keys)),
                upstream.sort_by_keys(pred, list(keys)),
            )["match"]
        )
    except Exception:  # a malformed submission is a failed one, not a crash
        return False


def score_model(
    name: str,
    gt: pd.DataFrame,
    pred: pd.DataFrame | None,
    keys: Sequence[str],
    baseline: float,
) -> ModelScore:
    """Score one data model against its ground truth.

    `baseline` is `weighted_baseline(gt, keys)`, precomputed once per model and
    cached — it depends only on the ground truth, never on the submission.

    Never raises: a failed rollout is data, not an error.

    An exception here would propagate out of the reward computation and take
    down every trajectory in the GRPO group, discarding good rollouts along
    with the bad one. Failures are recorded in `error` and scored zero.
    """
    if pred is None:
        return ModelScore(
            name, 0.0, baseline, 0.0, False, 0.0, 0.0, len(gt), 0,
            error="submitted table is missing",
        )

    score = weighted_cell_f1(gt, pred, keys)
    # A model a constant filler already solves carries no signal; the
    # correction would divide by ~zero, so nothing is awarded for it.
    reward = (
        0.0 if baseline >= 1.0
        else max(0.0, (score.f1 - baseline) / (1.0 - baseline))
    )

    return ModelScore(
        name=name,
        weighted_f1=score.f1,
        baseline=baseline,
        reward=reward,
        official_pass=official_verdict(gt, pred, keys),
        key_recall=score.alignment.key_recall,
        key_precision=score.alignment.key_precision,
        n_gt_rows=score.alignment.n_gt_rows,
        n_pred_rows=score.alignment.n_pred_rows,
        error=score.alignment.error,
    )


# --------------------------------------------------------------------------
# Information-weighted scoring
# --------------------------------------------------------------------------


def cell_weights(gt: pd.DataFrame, keys: Sequence[str]) -> dict[str, pd.Series]:
    """How much each ground-truth cell reveals, in bits.

    A value's weight is its self-information within its own column: a value
    filling most of the column is nearly free to guess and weighs almost
    nothing, while a rare one weighs a lot. A column that never varies weighs
    zero throughout, so reproducing it earns nothing — which is the point, since
    it says nothing about whether the pipeline was understood.

    On a table too small to estimate frequencies the weights flatten out and
    scoring degrades gracefully into the unweighted case.
    """
    weights: dict[str, pd.Series] = {}
    n_rows = len(gt)
    for column in scored_columns(gt, keys):
        if n_rows == 0:
            weights[column] = pd.Series(dtype=float)
            continue
        frequency = gt[column].value_counts(dropna=False, normalize=True)
        weights[column] = (
            gt[column].map(lambda v: -np.log2(frequency.get(v, 1.0 / n_rows)))
            .astype(float)
            .reset_index(drop=True)
        )
    return weights


def information_content(gt: pd.DataFrame, keys: Sequence[str]) -> float:
    """Total bits a correct submission would have to reproduce.

    Near zero means the model cannot distinguish understanding from guessing —
    every scored column is effectively constant — and it belongs out of the
    training pool rather than in it with an inflated score.
    """
    return float(sum(series.sum() for series in cell_weights(gt, keys).values()))


def weighted_cell_f1(
    gt: pd.DataFrame, pred: pd.DataFrame, keys: Sequence[str]
) -> CellScore:
    """Score a submission in bits recovered, discounted for producing the wrong
    number of rows.

    Two independent factors:

        recovery  =  hit / W                  share of the answer's bits returned
        size      =  2 |gt| / (|gt| + |t|)    1 when the row counts match,
                                              falling away in either direction
        score     =  recovery * size

    An earlier version tracked precision separately, charging each matched row
    the weight of the row it matched and each invented row the average. Across
    all 202 data models that changed 8% of scores by a mean of 0.001, so the
    closed form above replaced it. Where every value in a column is distinct
    the weights are uniform and this reduces exactly to `cell_f1`.
    """
    alignment = align_by_key(gt, pred, keys)
    columns = scored_columns(gt, keys)
    n_columns = len(columns)

    if alignment.error is not None or n_columns == 0:
        return CellScore(0.0, 0.0, 0.0, n_columns, 0, alignment)

    weights = cell_weights(gt, keys)
    total_weight = float(sum(series.sum() for series in weights.values()))
    if total_weight <= 0.0:
        # Nothing here varies, so nothing can be demonstrated by reproducing it.
        return CellScore(0.0, 0.0, 0.0, n_columns, 0, alignment)

    matched_weight = 0.0
    if alignment.n_matched:
        gt_rows = gt.iloc[list(alignment.gt_positions)]
        pred_rows = pred.iloc[list(alignment.pred_positions)]
        pred_by_lower = {str(c).lower(): c for c in pred.columns}
        for column in columns:
            submitted = pred_by_lower.get(str(column).lower())
            if submitted is None:
                continue  # column never produced; its bits stay unearned
            hits = np.array(cell_matches(gt_rows[column], pred_rows[submitted]))
            column_weights = weights[column].iloc[list(alignment.gt_positions)]
            matched_weight += float(column_weights.to_numpy()[hits].sum())

    recall = matched_weight / total_weight
    rows = alignment.n_gt_rows + alignment.n_pred_rows
    size_factor = 2.0 * alignment.n_gt_rows / rows if rows else 0.0
    # Reported for diagnostics; it is what the closed form implies, so the
    # usual harmonic identity between the three still holds.
    precision = (
        recall * alignment.n_gt_rows / alignment.n_pred_rows
        if alignment.n_pred_rows
        else 0.0
    )

    return CellScore(
        recall, precision, recall * size_factor, n_columns,
        alignment.n_matched, alignment,
    )


def weighted_baseline(gt: pd.DataFrame, keys: Sequence[str]) -> float:
    """What the best constant filler scores once cells are counted in bits.

    This is the zero point of the reward. Weighting drags it far below the
    unweighted equivalent — a median of 0.107 against 0.555 across the
    benchmark's 202 data models — which is what makes subtracting it safe:
    the leftover headroom is wide, so `1/(1-baseline)` magnifies differences by
    1.12x at the median and never by more than 2x, where the unweighted
    baseline reached 1000x on its worst model.
    """
    if gt.empty or not scored_columns(gt, keys):
        return 0.0  # no information to reproduce; see `information_content`
    return weighted_cell_f1(gt, constant_filled(gt, keys), keys).f1
