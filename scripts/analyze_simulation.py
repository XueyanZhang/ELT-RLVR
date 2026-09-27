"""Compare reward designs on the simulation output.

Four questions decide between them:

  separation  does every submission that did real work outscore every
              submission that did none? measured per model as the margin
              between the worst of the first group and the best of the second
  stability   does one policy score similarly across models, so a number means
              the same thing everywhere?
  coverage    how many models have to be thrown out as unscoreable?
  ordering    is a strictly better submission always scored higher?
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ["chance_corrected", "hybrid", "S1_simple_ratio", "S2_recall_penalty"]

# Submissions that required understanding the transformation, and ones that did not.
DID_WORK = ["perfect", "one_column_off", "half_rows"]
DID_NOTHING = ["constant_fill", "naive_constant", "keys_only", "empty"]

# Pairs where the left submission is unambiguously better than the right.
MUST_OUTRANK = [
    ("perfect", "one_column_off"),
    ("perfect", "half_rows"),
    ("perfect", "half_columns"),
    ("perfect", "duplicated"),
    ("perfect", "superset"),
    ("one_column_off", "constant_fill"),
    ("half_rows", "constant_fill"),
    ("constant_fill", "empty"),
    ("half_rows", "empty"),
    # The modelling choice we made deliberately, checked like any other rule.
    ("half_rows", "half_columns"),
]


def pivot(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    return frame.pivot_table(index=["db", "model"], columns="policy", values=variant)


def main() -> int:
    path = REPO / "data/reward_simulation.csv"
    frame = pd.read_csv(path)
    print(f"{frame['model'].nunique()} data models x {frame['policy'].nunique()} policies\n")

    print("=" * 74)
    print(f"{'policy':<18}" + "".join(f"{v:>20}" for v in VARIANTS))
    print("-" * 74)
    order = ["perfect", "hardcoded", "one_column_off", "half_rows", "half_columns",
             "duplicated", "superset", "constant_fill", "naive_constant", "keys_only", "empty"]
    for policy in order:
        rows = frame[frame.policy == policy]
        if rows.empty:
            continue
        cells = "".join(
            f"{rows[v].median():>9.3f} (sd {rows[v].std():.2f})" for v in VARIANTS
        )
        print(f"{policy:<18}{cells}")

    print("\n" + "=" * 74)
    print("SEPARATION  margin = worst real submission - best empty one, per model")
    print("-" * 74)
    for variant in VARIANTS:
        table = pivot(frame, variant)
        margin = table[DID_WORK].min(axis=1) - table[DID_NOTHING].max(axis=1)
        print(f"  {variant:<24} median {margin.median():+.3f}   "
              f"positive on {(margin > 0).mean():.0%} of models   "
              f"negative on {(margin < 0).sum()}")

    print("\n" + "=" * 74)
    print("STABILITY  spread of one policy's score across models (lower is better)")
    print("-" * 74)
    for variant in VARIANTS:
        spreads = {p: frame[frame.policy == p][variant].std() for p in order}
        worst = max(spreads, key=lambda p: spreads[p])
        print(f"  {variant:<24} mean sd {sum(spreads.values())/len(spreads):.3f}   "
              f"worst: {worst} ({spreads[worst]:.3f})")

    print("\n" + "=" * 74)
    print("COVERAGE  models that cannot be scored at all")
    print("-" * 74)
    for variant in VARIANTS:
        table = pivot(frame, variant)
        dead = (table["perfect"] <= 0).sum()
        flat = ((table["perfect"] - table["constant_fill"]).abs() < 0.05).sum()
        print(f"  {variant:<24} perfect scores 0 on {dead:>3}   "
              f"indistinguishable from guessing on {flat:>3}  of {len(table)}")

    print("\n" + "=" * 74)
    print("ORDERING  share of models where a better submission scores higher")
    print("-" * 74)
    print(f"{'pair':<40}" + "".join(f"{v:>20}" for v in VARIANTS))
    for better, worse in MUST_OUTRANK:
        cells = ""
        for variant in VARIANTS:
            table = pivot(frame, variant)
            holds = (table[better] >= table[worse] - 1e-9).mean()
            cells += f"{holds:>19.0%} "
        print(f"{better + ' >= ' + worse:<40}{cells}")

    print("\n" + "=" * 74)
    print("HARDCODING  a fabricated table scores exactly what a computed one does")
    print("-" * 74)
    for variant in VARIANTS:
        table = pivot(frame, variant)
        identical = (table["hardcoded"] - table["perfect"]).abs().max()
        print(f"  {variant:<24} largest gap between them: {identical:.6f}"
              "   <- no value comparison can tell them apart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
