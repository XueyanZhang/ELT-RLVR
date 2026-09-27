"""The cell-level comparison predicate must not drift from the official one.

Our reward scores individual cells so a partially correct data model earns
partial credit; the official comparator only judges whole columns. These tests
pin the relationship: for equal-length columns, "every cell matches" must mean
exactly what the official comparator calls a match.

Note the comparison MODE is chosen per column, not per cell: the official
comparator only compares numerically when the whole column coerces to numbers.
`cell_matches` therefore takes both columns, never a bare pair of values.
"""
import pandas as pd

from elt_rlvr.reward import cell_matches


def test_all_cells_match_when_official_says_column_matches(official, gt_csv_paths):
    """Real ground truth compared against itself: both verdicts must agree."""
    checked = 0
    for path in gt_csv_paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
        if df.empty:
            continue
        for col in df.columns:
            series = df[col]
            ours = all(cell_matches(series, series))
            theirs = official._vectors_match(series, series)
            assert ours == theirs, f"{path.name}:{col} ours={ours} theirs={theirs}"
            checked += 1
    assert checked > 100, f"only exercised {checked} columns"


def test_mode_is_chosen_per_column_not_per_cell(official):
    """A non-numeric cell forces the whole column onto the string branch.

    "1" and "1.0" are numerically equal but textually different, so they match
    in a fully numeric column and clash in a mixed one. Deciding per cell would
    silently disagree with the official verdict.
    """
    mixed_gt, mixed_pred = pd.Series(["1", "abc"]), pd.Series(["1.0", "abc"])
    assert official._vectors_match(mixed_gt, mixed_pred) is False
    assert list(cell_matches(mixed_gt, mixed_pred)) == [False, True]

    numeric_gt, numeric_pred = pd.Series(["1", "2"]), pd.Series(["1.0", "2.0"])
    assert official._vectors_match(numeric_gt, numeric_pred) is True
    assert list(cell_matches(numeric_gt, numeric_pred)) == [True, True]


def test_one_wrong_cell_is_localised(official):
    """A single bad cell fails only that cell, though the column verdict flips."""
    gt, pred = pd.Series(["10", "20", "30"]), pd.Series(["10", "999", "30"])
    assert official._vectors_match(gt, pred) is False
    assert list(cell_matches(gt, pred)) == [True, False, True]


def test_null_matches_null_but_not_a_value(official):
    """Inherited from the official comparator: NaN == NaN, NaN != value."""
    gt = pd.Series(["a", None, None])
    pred = pd.Series(["a", None, "x"])
    assert official._vectors_match(gt, pred) is False
    assert list(cell_matches(gt, pred)) == [True, True, False]
