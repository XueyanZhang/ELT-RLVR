"""Scoring cells by how much they actually reveal.

Chance correction subtracts one number per data model — what a constant filler
would score. That is too coarse: within a single model some columns never vary
and carry nothing, while others hold everything. Weighting each cell by its
self-information spends the credit where the information is, and needs no
correction term afterwards, so there is no `1/(1-b)` factor to amplify noise
on models that are mostly predictable.
"""
import math

import pytest

from elt_rlvr.reward import (
    cell_f1,
    cell_weights,
    information_content,
    weighted_cell_f1,
)
from tests.conftest import frame


KEYS = ["id"]


def test_a_column_that_never_varies_is_worth_nothing():
    """If every row holds the same value, reproducing it proves nothing."""
    gt = frame(id=[1, 2, 3, 4], constant=["x"] * 4)

    assert list(cell_weights(gt, KEYS)["constant"]) == [0.0, 0.0, 0.0, 0.0]


def test_rare_values_outweigh_common_ones():
    """The weight is self-information, so the minority value carries more."""
    gt = frame(id=[1, 2, 3, 4], mostly_zero=[0, 0, 0, 7])

    weights = cell_weights(gt, KEYS)["mostly_zero"]

    assert weights.iloc[0] == pytest.approx(-math.log2(3 / 4))
    assert weights.iloc[3] == pytest.approx(-math.log2(1 / 4))
    assert weights.iloc[3] > weights.iloc[0] * 4


def test_weighting_reduces_to_plain_scoring_when_every_value_is_unique():
    """Uniform weights must not change any verdict — the ordering we pinned for
    partial output still has to hold once weights are introduced."""
    gt = frame(id=[1, 2, 3, 4], a=["p", "q", "r", "s"], b=["w", "x", "y", "z"])
    half_the_rows = gt.iloc[:2]
    every_row_half_wrong = frame(
        id=[1, 2, 3, 4], a=["p", "q", "r", "s"], b=["!"] * 4
    )

    for pred in (gt, half_the_rows, every_row_half_wrong, gt.iloc[:0]):
        assert weighted_cell_f1(gt, pred, KEYS).f1 == pytest.approx(
            cell_f1(gt, pred, KEYS).f1
        )


def test_getting_only_the_common_value_right_scores_far_lower_than_unweighted():
    """The sparse-table case: a column that is 75% zeros.

    Plain scoring pays 0.75 for filling in the majority value. Weighted scoring
    pays only for the information that majority actually carried.
    """
    gt = frame(id=[1, 2, 3, 4], v=[0, 0, 0, 7])
    all_zeros = frame(id=[1, 2, 3, 4], v=[0, 0, 0, 0])

    assert cell_f1(gt, all_zeros, KEYS).f1 == pytest.approx(0.75)
    assert weighted_cell_f1(gt, all_zeros, KEYS).f1 < 0.4


def test_finding_the_rare_value_earns_most_of_the_credit():
    gt = frame(id=[1, 2, 3, 4], v=[0, 0, 0, 7])
    only_the_outlier = frame(id=[1, 2, 3, 4], v=["!", "!", "!", 7])

    common_only = weighted_cell_f1(gt, frame(id=[1, 2, 3, 4], v=[0, 0, 0, 0]), KEYS)
    rare_only = weighted_cell_f1(gt, only_the_outlier, KEYS)

    assert rare_only.f1 > common_only.f1


def test_a_correct_submission_still_scores_one():
    gt = frame(id=[1, 2, 3, 4], v=[0, 0, 0, 7], w=["a", "a", "b", "c"])

    assert weighted_cell_f1(gt, gt, KEYS).f1 == 1.0


def test_padding_with_junk_rows_still_costs_precision():
    gt = frame(id=[1, 2], v=[5, 9])
    padded = frame(id=[1, 2, 3, 4], v=[5, 9, 0, 0])

    assert weighted_cell_f1(gt, padded, KEYS).f1 < weighted_cell_f1(gt, gt, KEYS).f1


def test_a_model_carrying_no_information_is_reported_as_such():
    """Every scored column constant: no submission can demonstrate understanding."""
    gt = frame(id=[1, 2, 3], a=["same"] * 3, b=["0"] * 3)

    score = weighted_cell_f1(gt, gt, KEYS)

    assert information_content(gt, KEYS) == 0.0
    assert score.f1 == 0.0  # nothing to earn, so nothing is awarded


def test_information_content_ranks_real_models_the_way_we_measured(gt_csv_paths):
    """The degenerate fixtures found by measurement must come out at the bottom."""
    import json

    import pandas as pd

    from tests.conftest import OFFICIAL_EVAL_DIR

    sort_keys = json.loads((OFFICIAL_EVAL_DIR / "sort_key.json").read_text())
    per_cell = {}
    for path in gt_csv_paths:
        gt = pd.read_csv(path, dtype=str, keep_default_na=True)
        keys = sort_keys.get(path.parent.name, {}).get(path.stem, [])
        if gt.empty or not keys:
            continue
        cells = max(len(gt) * len(gt.columns), 1)
        per_cell[f"{path.parent.name}.{path.stem}"] = information_content(gt, keys) / cells

    assert per_cell["twitter_organic.twitter_organic__tweets"] == 0.0
    assert per_cell["github.github__issues"] > 1.0


def test_score_separates_into_recovery_and_a_size_penalty():
    """The scoring rule is two independent factors, not a precision term.

    An earlier version charged each matched row the weight of the ground-truth
    row it matched and each invented row the average. Measured across 202
    models that extra machinery moved 8% of scores by a mean of 0.001, so it
    was dropped in favour of the closed form asserted here.

    The weights must be uneven for this to discriminate: where every value is
    unique the two formulations agree exactly.
    """
    gt = frame(id=[1, 2, 3, 4], v=["x", "x", "x", "y"])
    half = gt.iloc[:2]

    score = weighted_cell_f1(gt, half, KEYS)

    expected = 2 * score.recall * len(gt) / (len(gt) + len(half))
    assert score.f1 == pytest.approx(expected)


def test_producing_the_right_number_of_rows_incurs_no_penalty():
    """The size factor `2|gt| / (|gt| + |t|)` is 1 exactly when the row counts
    match, so a correct submission is scored on recovery alone."""
    gt = frame(id=[1, 2, 3, 4], v=["x", "x", "x", "y"])
    wrong_values = frame(id=[1, 2, 3, 4], v=["!", "!", "!", "!"])

    for pred in (gt, wrong_values):
        score = weighted_cell_f1(gt, pred, KEYS)
        assert score.f1 == pytest.approx(score.recall)
