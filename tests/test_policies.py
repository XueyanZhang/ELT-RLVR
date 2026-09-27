"""Synthetic submissions standing in for behaviours a policy might learn.

These are the yardstick the reward is measured against: a reward function is
only useful if it separates a pipeline that was understood from one that was
guessed, and the only way to know whether ours does is to feed it both and
look. They double as the adversarial suite — each shortcut is written out as
code and asserted to earn little.
"""
import pandas as pd

from elt_rlvr import policies
from elt_rlvr.reward import scored_columns
from tests.conftest import frame


GT = frame(id=[1, 2, 3, 4], v=[0, 0, 0, 7], w=["a", "a", "b", "c"])
KEYS = ["id"]


def test_perfect_returns_the_answer_unchanged():
    assert policies.perfect(GT, KEYS).equals(GT)


def test_empty_keeps_the_schema_but_no_rows():
    result = policies.empty(GT, KEYS)

    assert list(result.columns) == list(GT.columns)
    assert len(result) == 0


def test_keys_only_gets_the_row_set_right_and_nothing_else():
    result = policies.keys_only(GT, KEYS)

    assert len(result) == len(GT)
    assert list(result["id"]) == list(GT["id"])
    assert result["v"].isna().all()


def test_constant_fill_uses_each_columns_most_common_value():
    result = policies.constant_fill(GT, KEYS)

    assert set(result["v"]) == {"0"}  # the majority value
    assert set(result["w"]) == {"a"}


def test_naive_constant_fills_zero_everywhere():
    """What an untrained policy actually emits, as opposed to an oracle's mode."""
    result = policies.naive_constant(GT, KEYS)

    assert set(result["v"]) == {"0"} and set(result["w"]) == {"0"}


def test_half_rows_keeps_whole_rows_correct():
    result = policies.half_rows(GT, KEYS)

    assert len(result) == 2
    assert result.reset_index(drop=True).equals(GT.iloc[:2].reset_index(drop=True))


def test_half_columns_keeps_every_row_but_corrupts_half_the_values():
    result = policies.half_columns(GT, KEYS)

    assert len(result) == len(GT)
    corrupted = [c for c in scored_columns(GT, KEYS) if not result[c].equals(GT[c])]
    assert len(corrupted) == 1  # one of the two scored columns


def test_duplicated_repeats_every_row():
    result = policies.duplicated(GT, KEYS, times=3)

    assert len(result) == 3 * len(GT)


def test_superset_adds_rows_whose_keys_are_not_in_the_answer():
    result = policies.superset(GT, KEYS)

    assert len(result) > len(GT)
    assert set(GT["id"]).issubset(set(result["id"]))
    assert len(set(result["id"]) - set(GT["id"])) > 0


def test_one_column_off_corrupts_exactly_one_column():
    result = policies.one_column_off(GT, KEYS)

    differing = [c for c in scored_columns(GT, KEYS) if not result[c].equals(GT[c])]
    assert len(differing) == 1


def test_hardcoded_is_byte_identical_to_the_answer():
    """Stated as a test because it is the whole problem: no comparison of
    values can tell a fabricated table from a computed one. Only re-running the
    pipeline against altered inputs can."""
    assert policies.hardcoded(GT, KEYS).equals(policies.perfect(GT, KEYS))


def test_every_policy_runs_on_real_ground_truth(gt_csv_paths):
    import json

    from tests.conftest import OFFICIAL_EVAL_DIR

    sort_keys = json.loads((OFFICIAL_EVAL_DIR / "sort_key.json").read_text())
    for path in gt_csv_paths[:12]:
        gt = pd.read_csv(path, dtype=str, keep_default_na=True)
        keys = sort_keys.get(path.parent.name, {}).get(path.stem, [])
        if gt.empty or not keys:
            continue
        for name, policy in policies.ALL.items():
            result = policy(gt, keys)
            assert isinstance(result, pd.DataFrame), f"{name} on {path.stem}"
