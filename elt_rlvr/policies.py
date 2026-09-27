"""Stand-in submissions used to characterise the reward function.

A reward is only useful if it tells apart a pipeline that was understood from
one that was guessed. Rather than argue about that, we hand it submissions
whose quality we already know — correct, empty, padded, filled with the most
common value — and look at what comes out. The shortcuts double as an
adversarial suite: each is a way of scoring without doing the work, written out
so it can be asserted to fail.

Every policy takes the ground truth and the model's key columns and returns a
table shaped like a submission. Several of them are *given* the answer, which
no real policy would be; that is deliberate, since the question is what the
reward pays for, not what a policy could discover.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from elt_rlvr.reward import scored_columns

# A value chosen to match nothing: not numeric, not a real string in any column.
WRONG = "\x01wrong"


def perfect(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The answer. The only submission that should score one."""
    return gt.copy()


def hardcoded(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The answer again — written into the table instead of computed.

    Identical to `perfect` by construction, which is exactly the finding: no
    amount of comparing values can separate the two. Catching it needs the
    pipeline re-run against altered inputs.
    """
    return gt.copy()


def empty(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The schema, and nothing in it."""
    return gt.iloc[:0].copy()


def keys_only(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The right rows, with every other column left null.

    Reachable by selecting distinct keys off a source table, so whatever this
    scores is available without understanding the transformation.
    """
    result = gt.copy()
    for column in scored_columns(gt, keys):
        result[column] = pd.NA
    return result


def constant_fill(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The right rows, each column filled with its own most common value.

    An oracle's version of guessing: no real policy knows the mode, so this is
    the ceiling on what filling in a constant can earn.
    """
    result = gt.copy()
    for column in scored_columns(gt, keys):
        modes = gt[column].mode(dropna=False)
        result[column] = modes.iloc[0] if len(modes) else pd.NA
    return result


def naive_constant(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The right rows, everything else zero — what an untrained policy emits."""
    result = gt.copy()
    for column in scored_columns(gt, keys):
        result[column] = "0"
    return result


def half_rows(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Half the rows, each of them entirely correct.

    A restrictive join or an over-tight filter looks like this.
    """
    return gt.iloc[: len(gt) // 2].copy()


def half_columns(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Every row, with half of the value columns wrong.

    The right grain with the wrong arithmetic. Scores the same number of
    correct cells as `half_rows`; which of the two ranks higher is the one
    modelling choice the reward makes on its own.
    """
    result = gt.copy()
    columns = scored_columns(gt, keys)
    for column in columns[: max(1, len(columns) // 2)]:
        result[column] = WRONG
    return result


def one_column_off(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Correct but for a single column — a near miss, to check sensitivity."""
    result = gt.copy()
    columns = scored_columns(gt, keys)
    if columns:
        result[columns[0]] = WRONG
    return result


def duplicated(gt: pd.DataFrame, keys: Sequence[str], times: int = 3) -> pd.DataFrame:
    """Every row repeated. A missing DISTINCT, or a fan-out join."""
    return pd.concat([gt] * times, ignore_index=True)


def superset(gt: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """The answer plus invented rows — the shape a cross join leaves behind."""
    if gt.empty:
        return gt.copy()
    extra = gt.copy()
    for key in keys:
        column = _resolve(key, gt.columns)
        if column is not None:
            extra[column] = [f"{value}\x02x" for value in extra[column]]
    return pd.concat([gt, extra], ignore_index=True)


def _resolve(key: str, columns) -> str | None:
    from elt_rlvr.reward import resolve_key

    return resolve_key(key, columns)


Policy = Callable[[pd.DataFrame, Sequence[str]], pd.DataFrame]

#: Ordered roughly best to worst, so result tables read top to bottom.
ALL: dict[str, Policy] = {
    "perfect": perfect,
    "hardcoded": hardcoded,
    "one_column_off": one_column_off,
    "half_rows": half_rows,
    "half_columns": half_columns,
    "duplicated": duplicated,
    "superset": superset,
    "constant_fill": constant_fill,
    "naive_constant": naive_constant,
    "keys_only": keys_only,
    "empty": empty,
}
