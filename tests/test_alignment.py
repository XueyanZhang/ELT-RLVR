"""Aligning a submitted table against ground truth by unique key.

Row order carries no meaning — a pipeline that produces the right rows in a
different order is correct — so rows are paired by the model's unique key
before any value is compared. Everything downstream (recall, precision, cell
scoring) depends on this pairing being right.
"""
import pandas as pd
import pytest

from elt_rlvr.reward import align_by_key, resolve_key


def frame(**cols) -> pd.DataFrame:
    return pd.DataFrame({k: [str(x) for x in v] for k, v in cols.items()}, dtype=str)


def test_aligns_rows_by_key_ignoring_row_order():
    gt = frame(id=[1, 2, 3], v=["a", "b", "c"])
    pred = frame(id=[3, 1, 2], v=["c", "a", "b"])

    alignment = align_by_key(gt, pred, ["id"])

    assert alignment.error is None
    assert alignment.n_matched == 3
    assert alignment.key_recall == 1.0
    assert alignment.key_precision == 1.0
    # Pairing must line up the same key, not the same position.
    for g, p in zip(alignment.gt_positions, alignment.pred_positions):
        assert gt.loc[g, "id"] == pred.loc[p, "id"]


def test_key_column_names_match_case_insensitively():
    """Warehouses differ on identifier case; the benchmark folds it too."""
    gt = frame(UNIQUE_SESSION_ID=[1, 2], v=["a", "b"])
    pred = frame(unique_session_id=[1, 2], v=["a", "b"])

    assert align_by_key(gt, pred, ["UNIQUE_SESSION_ID"]).n_matched == 2


def test_cast_expression_key_resolves_to_its_column():
    """13 of the benchmark's keys are written as `cast(col as integer)`."""
    assert resolve_key("cast(driver_id as integer)", ["driver_id", "name"]) == "driver_id"
    assert resolve_key("CAST(label_id AS INT)", ["label_id"]) == "label_id"


def test_nulls_first_expression_key_resolves_to_its_column():
    assert resolve_key("video_id NULLS FIRST", ["video_id"]) == "video_id"


def test_unparseable_key_expression_is_reported_not_ignored():
    """Silently falling back to positional order would make scores order-dependent."""
    assert resolve_key("coalesce(a, b)", ["a", "b"]) is None

    gt = frame(a=[1], b=[2])
    alignment = align_by_key(gt, frame(a=[1], b=[2]), ["coalesce(a, b)"])
    assert alignment.error is not None
    assert "coalesce(a, b)" in alignment.error
    assert alignment.n_matched == 0


def test_duplicate_prediction_keys_match_at_most_one_ground_truth_row():
    """Padding the output with copies must cost precision, not buy recall."""
    gt = frame(id=[1, 2], v=["a", "b"])
    pred = frame(id=[1, 1, 1, 2], v=["a", "a", "a", "b"])

    alignment = align_by_key(gt, pred, ["id"])

    assert alignment.n_matched == 2
    assert alignment.key_recall == 1.0
    assert alignment.key_precision == 0.5  # 2 of 4 predicted rows are useful


def test_missing_key_column_in_prediction_is_reported():
    gt = frame(id=[1], v=["a"])
    alignment = align_by_key(gt, frame(v=["a"]), ["id"])

    assert alignment.error is not None
    assert alignment.n_matched == 0


def test_composite_key_aligns_on_all_parts():
    gt = frame(day=["d1", "d1", "d2"], repo=["r1", "r2", "r1"], v=[1, 2, 3])
    pred = frame(day=["d2", "d1", "d1"], repo=["r1", "r2", "r1"], v=[3, 2, 1])

    alignment = align_by_key(gt, pred, ["day", "repo"])

    assert alignment.n_matched == 3
    for g, p in zip(alignment.gt_positions, alignment.pred_positions):
        assert (gt.loc[g, "day"], gt.loc[g, "repo"]) == (pred.loc[p, "day"], pred.loc[p, "repo"])


def test_empty_ground_truth_does_not_crash():
    alignment = align_by_key(frame(id=[]), frame(id=[]), ["id"])

    assert alignment.error is None
    assert alignment.n_matched == 0
    assert alignment.key_recall == 0.0


@pytest.mark.parametrize("missing_side", ["gt", "pred"])
def test_unmatched_rows_reduce_the_right_ratio(missing_side):
    full, partial = frame(id=[1, 2, 3, 4]), frame(id=[1, 2])
    gt, pred = (full, partial) if missing_side == "gt" else (partial, full)

    alignment = align_by_key(gt, pred, ["id"])

    assert alignment.n_matched == 2
    if missing_side == "gt":
        assert alignment.key_recall == 0.5 and alignment.key_precision == 1.0
    else:
        assert alignment.key_recall == 1.0 and alignment.key_precision == 0.5


def test_every_ground_truth_table_aligns_perfectly_with_itself(gt_csv_paths):
    """End-to-end guard over real benchmark data.

    The unit tests above pin each rule in isolation with hand-built frames.
    This one runs the whole resolution-and-pairing path over every fixture,
    including the four models whose keys are SQL fragments rather than column
    names, where an unnoticed regression would silently score correct output
    as wrong.
    """
    import json

    from tests.conftest import OFFICIAL_EVAL_DIR

    sort_keys = json.loads((OFFICIAL_EVAL_DIR / "sort_key.json").read_text())

    for path in gt_csv_paths:
        database, table = path.parent.name, path.stem
        frame = pd.read_csv(path, dtype=str, keep_default_na=True)
        keys = sort_keys.get(database, {}).get(table, [])

        alignment = align_by_key(frame, frame, keys)

        assert alignment.error is None, f"{database}.{table}: {alignment.error}"
        assert alignment.key_recall == 1.0, f"{database}.{table} recall"
        assert alignment.key_precision == 1.0, f"{database}.{table} precision"
