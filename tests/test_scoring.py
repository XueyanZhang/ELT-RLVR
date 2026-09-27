"""Turning a submitted table into a training signal.

`cell_f1` and `chance_baseline` are the unweighted design, kept because
`scripts/simulate_reward.py` measures it against the one we shipped; the reward
the environment returns is built in `test_hybrid_reward.py`.

Two numbers come out of every data model: the benchmark's own pass/fail, which
we report, and a graded score, which we train on. The graded score exists
because the pass/fail cannot train anything — see `reward.py`.

The score is chance-corrected: a model whose ground truth is mostly constant
columns can be half-filled by a policy that understands nothing, and measuring
real ground truth showed that is the common case, not a corner case. Zero means
"no better than filling each column with its most common value"; one means
correct.
"""
import pytest

from elt_rlvr.reward import cell_f1, chance_baseline, score_model, weighted_baseline
from tests.conftest import frame


# Four rows, a key and two scored columns — small enough to verify by hand.
GT = frame(id=[1, 2, 3, 4], a=["p", "q", "r", "s"], b=["w", "x", "y", "z"])
KEYS = ["id"]


def test_perfect_submission_scores_one():
    assert cell_f1(GT, GT, KEYS).f1 == 1.0


def test_empty_submission_scores_zero():
    assert cell_f1(GT, GT.iloc[:0], KEYS).f1 == 0.0


def test_key_columns_are_not_scored():
    """Keys align the rows, so crediting them would be circular."""
    score = cell_f1(GT, GT, KEYS)
    assert score.n_scored_columns == 2  # a and b, not id

    # Right keys, both scored columns wrong: nothing earned.
    keys_only = frame(id=[1, 2, 3, 4], a=["!"] * 4, b=["!"] * 4)
    assert cell_f1(GT, keys_only, KEYS).f1 == 0.0


def test_partial_output_beats_complete_but_wrong_output():
    """A deliberate modelling choice, pinned here so it cannot drift silently.

    Producing half the rows correctly and producing every row with half its
    values wrong yield the same number of correct cells. We rank the first
    higher: cells that were emitted but are wrong count against precision,
    whereas cells never emitted only cost recall. Both orderings are defensible
    — see the design doc — and nothing in the data decides between them.
    """
    half_the_rows = GT.iloc[:2]
    every_row_half_wrong = frame(
        id=[1, 2, 3, 4], a=["p", "q", "r", "s"], b=["!", "!", "!", "!"]
    )

    partial = cell_f1(GT, half_the_rows, KEYS)
    complete_but_wrong = cell_f1(GT, every_row_half_wrong, KEYS)

    assert partial.recall == complete_but_wrong.recall == 0.5
    assert partial.precision == 1.0 and complete_but_wrong.precision == 0.5
    assert partial.f1 == pytest.approx(2 / 3)
    assert complete_but_wrong.f1 == pytest.approx(0.5)
    assert partial.f1 > complete_but_wrong.f1


def test_duplicated_rows_cost_precision():
    """Padding the output cannot buy recall; it only dilutes precision."""
    padded = frame(
        id=[1, 1, 2, 2, 3, 3, 4, 4],
        a=["p", "p", "q", "q", "r", "r", "s", "s"],
        b=["w", "w", "x", "x", "y", "y", "z", "z"],
    )

    score = cell_f1(GT, padded, KEYS)

    assert score.recall == 1.0
    assert score.precision == 0.5
    assert score.f1 < cell_f1(GT, GT, KEYS).f1


def test_dropping_columns_does_not_inflate_precision():
    """Precision is measured per expected cell, so emitting fewer columns
    leaves the missing ones counting as unearned rather than unasked."""
    one_column = frame(id=[1, 2, 3, 4], a=["p", "q", "r", "s"])

    score = cell_f1(GT, one_column, KEYS)

    assert score.precision == 0.5  # 4 of 8 expected cells, not 4 of 4 emitted


def test_constant_ground_truth_is_fully_predictable():
    """Every scored column holds one value, so a constant filler is perfect."""
    constant = frame(id=[1, 2, 3], a=["same"] * 3, b=["0"] * 3)

    assert chance_baseline(constant, KEYS) == 1.0


def test_varied_ground_truth_is_not_predictable():
    assert chance_baseline(GT, KEYS) == pytest.approx(0.25)  # best constant hits 1 of 4


def test_official_verdict_is_reported_alongside_the_training_score():
    """Double-track accounting: we train on the graded score and report theirs."""
    baseline = weighted_baseline(GT, KEYS)
    wrong = frame(id=[1, 2, 3, 4], a=["p", "q", "r", "s"], b=["!"] * 4)

    assert score_model("m", GT, GT, KEYS, baseline).official_pass is True
    assert score_model("m", GT, wrong, KEYS, baseline).official_pass is False


def test_a_missing_table_is_scored_not_raised():
    """Rollouts fail often; an exception here would abort the whole GRPO group."""
    result = score_model("m", GT, None, KEYS, weighted_baseline(GT, KEYS))

    assert result.reward == 0.0
    assert result.error is not None


def test_real_ground_truth_is_largely_guessable(gt_csv_paths):
    """Measured, not assumed: the small-task pool is full of constant columns."""
    import json

    import pandas as pd

    from tests.conftest import OFFICIAL_EVAL_DIR

    sort_keys = json.loads((OFFICIAL_EVAL_DIR / "sort_key.json").read_text())
    baselines = []
    for path in gt_csv_paths:
        gt = pd.read_csv(path, dtype=str, keep_default_na=True)
        keys = sort_keys.get(path.parent.name, {}).get(path.stem, [])
        if gt.empty or not keys:
            continue
        baselines.append(chance_baseline(gt, keys))

    assert len(baselines) > 50
    assert sum(baselines) / len(baselines) > 0.2, "constant-filling should score well"
    assert max(baselines) > 0.9, "some models are almost entirely predictable"
