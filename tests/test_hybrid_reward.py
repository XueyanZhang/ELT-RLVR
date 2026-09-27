"""The reward the environment actually returns.

Measured across all 202 data models, weighting cells by information and then
subtracting what a constant filler earns beat both halves on their own:

                              chance-corrected   weighted   both
    separation (median)             +0.235        +0.505   +0.594
    partial work scored ~0        288/606         24/606    45/606
    worst noise amplification       1000x          1.00x     2.0x

Weighting is what keeps the correction harmless — it drags the constant
filler's score from a median of 0.555 down to 0.107, so dividing by the
remaining headroom barely magnifies anything. Correction is what keeps the
weighting honest: on its own it left three models where guessing outscored
working.
"""
import pytest

from elt_rlvr.reward import score_model, weighted_baseline
from elt_rlvr import policies
from tests.conftest import frame


GT = frame(id=[1, 2, 3, 4], v=[0, 0, 0, 7], w=["a", "a", "b", "c"])
KEYS = ["id"]


def reward_for(policy, gt=GT, keys=KEYS) -> float:
    return score_model("m", gt, policy(gt, keys), keys, weighted_baseline(gt, keys)).reward


def test_a_correct_submission_scores_one():
    assert reward_for(policies.perfect) == pytest.approx(1.0)


def test_filling_in_the_most_common_value_earns_nothing():
    """Zero is defined as "no better than the best possible guess"."""
    assert reward_for(policies.constant_fill) == 0.0


def test_cruder_guesses_earn_nothing_either():
    assert reward_for(policies.naive_constant) == 0.0
    assert reward_for(policies.keys_only) == 0.0
    assert reward_for(policies.empty) == 0.0


def test_partial_work_outscores_guessing():
    """The property chance correction alone failed: half the rows, done right,
    must not collapse to zero, or a group of partial rollouts scores uniformly
    and GRPO discards the whole group."""
    assert reward_for(policies.half_rows) > reward_for(policies.constant_fill)
    assert reward_for(policies.half_rows) > 0.0
    assert reward_for(policies.one_column_off) > reward_for(policies.half_rows)


def test_partial_work_keeps_substantial_credit_on_real_models(gt_csv_paths):
    """How much credit survives is a question about real ground truth, not
    about a four-row example, where a constant filler does unusually well."""
    import json
    import statistics

    import pandas as pd

    from tests.conftest import OFFICIAL_EVAL_DIR

    sort_keys = json.loads((OFFICIAL_EVAL_DIR / "sort_key.json").read_text())
    rewards = []
    for path in gt_csv_paths:
        gt = pd.read_csv(path, dtype=str, keep_default_na=True)
        keys = sort_keys.get(path.parent.name, {}).get(path.stem, [])
        if gt.empty or not keys:
            continue
        rewards.append(reward_for(policies.half_rows, gt, keys))

    assert statistics.median(rewards) > 0.4
    assert sum(r > 0.2 for r in rewards) / len(rewards) > 0.8


def test_padding_scores_below_the_answer_it_pads():
    assert reward_for(policies.duplicated) < reward_for(policies.perfect)
    assert reward_for(policies.superset) < reward_for(policies.perfect)


def test_reward_stays_within_unit_range(gt_csv_paths):
    import json

    import pandas as pd

    from tests.conftest import OFFICIAL_EVAL_DIR

    sort_keys = json.loads((OFFICIAL_EVAL_DIR / "sort_key.json").read_text())
    for path in gt_csv_paths[:15]:
        gt = pd.read_csv(path, dtype=str, keep_default_na=True)
        keys = sort_keys.get(path.parent.name, {}).get(path.stem, [])
        if gt.empty or not keys:
            continue
        baseline = weighted_baseline(gt, keys)
        for name, policy in policies.ALL.items():
            result = score_model("m", gt, policy(gt, keys), keys, baseline)
            assert 0.0 <= result.reward <= 1.0, f"{path.stem}/{name}: {result.reward}"


def test_noise_amplification_stays_small_on_real_models(gt_csv_paths):
    """`1/(1-baseline)` is the factor by which a small scoring difference is
    magnified. Under unweighted scoring it reached 1000x on one model."""
    import json

    import pandas as pd

    from tests.conftest import OFFICIAL_EVAL_DIR

    sort_keys = json.loads((OFFICIAL_EVAL_DIR / "sort_key.json").read_text())
    factors = []
    for path in gt_csv_paths:
        gt = pd.read_csv(path, dtype=str, keep_default_na=True)
        keys = sort_keys.get(path.parent.name, {}).get(path.stem, [])
        if gt.empty or not keys:
            continue
        baseline = weighted_baseline(gt, keys)
        if baseline < 1.0:
            factors.append(1.0 / (1.0 - baseline))

    assert max(factors) < 3.0, f"worst amplification {max(factors):.1f}x"


def test_a_model_with_nothing_to_learn_scores_zero_even_when_correct():
    """Every column constant: a perfect submission proves nothing, so the
    model belongs out of the training pool rather than in it scoring one."""
    degenerate = frame(id=[1, 2, 3], a=["x"] * 3, b=["0"] * 3)

    assert weighted_baseline(degenerate, KEYS) == 0.0
    assert reward_for(policies.perfect, degenerate) == 0.0
