"""Characterise reward variants by scoring submissions of known quality.

Runs every synthetic policy against every data model's ground truth under both
reward designs, so the choice between them rests on measured behaviour rather
than argument. Writes one row per (model, policy, variant) to a CSV.

    python scripts/simulate_reward.py [--max-rows N] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elt_rlvr import policies  # noqa: E402
from elt_rlvr.reward import (  # noqa: E402
    cell_f1,
    constant_filled,
    chance_baseline,
    information_content,
    official_verdict,
    scored_columns,
    weighted_cell_f1,
)

REPO = Path(__file__).resolve().parents[1]
GT_ROOT = REPO / "data/gt/gt_snowflake"
SORT_KEYS = json.loads((REPO / "third_party/ELT-Bench/evaluation/sort_key.json").read_text())


def model_context(gt, keys) -> dict:
    """Per-model quantities every variant needs, computed once rather than per
    policy: they depend only on the ground truth."""
    guess = constant_filled(gt, keys)
    guess_score = weighted_cell_f1(gt, guess, keys)
    n_gt = len(gt)
    return {
        "unweighted_floor": chance_baseline(gt, keys),
        "weighted_floor": guess_score.f1,
        "guess_recall": guess_score.recall,
        "s1_floor": _s1_from(guess_score.recall, n_gt, len(guess)),
        "n_gt": n_gt,
    }


def _s1_from(recall: float, n_gt: int, n_pred: int) -> float:
    total = n_gt + n_pred
    return 2.0 * recall * n_gt / total if total else 0.0


def _lift(score: float, floor: float) -> float:
    """How far above the guessing floor, as a share of the headroom left."""
    if floor >= 1.0:
        return 0.0
    return max(0.0, (score - floor) / (1.0 - floor))


def chance_corrected(scored, ctx) -> float:
    """Variant A: the benchmark's cell agreement, less what guessing earns."""
    return _lift(scored["plain"].f1, ctx["unweighted_floor"])


def information_weighted(scored, ctx) -> float:
    """Variant B: cells counted in bits, so guessable ones pay little."""
    return scored["weighted"].f1


def hybrid(scored, ctx) -> float:
    """Variant C, currently shipped: weighted cells, less what guessing earns.

    Weighting drops the guessing baseline far enough that subtracting it costs
    almost no headroom, which is why the two compose rather than compete.
    """
    return _lift(scored["weighted"].f1, ctx["weighted_floor"])


# ---- candidate simplifications of `hybrid`, measured before adopting one ----
#
# `hybrid` charges each matched row the weight of the ground-truth row it
# matched, and each invented row the average. Both simplifications below drop
# that distinction; the question is whether anything measurable is lost.


def simple_ratio(scored, ctx) -> float:
    """S1: charge every submitted row the average row's worth of bits.

    The recall / precision / harmonic-mean construction then collapses into
    `2 * recall * |gt| / (|gt| + |t|)` - one line instead of six.
    """
    score = scored["weighted"]
    s1 = _s1_from(score.recall, ctx["n_gt"], score.alignment.n_pred_rows)
    return _lift(s1, ctx["s1_floor"])


def recall_times_penalty(scored, ctx) -> float:
    """S2: no F1 at all - information recovered beyond guessing, discounted for
    over-production."""
    score = scored["weighted"]
    recovered = _lift(score.recall, ctx["guess_recall"])
    n_pred = score.alignment.n_pred_rows
    padding = min(1.0, ctx["n_gt"] / n_pred) if n_pred else 0.0
    return recovered * padding


VARIANTS = {
    "chance_corrected": chance_corrected,
    "information_weighted": information_weighted,
    "hybrid": hybrid,
    "S1_simple_ratio": simple_ratio,
    "S2_recall_penalty": recall_times_penalty,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=20_000,
                        help="truncate larger ground truth; keeps the sweep tractable")
    parser.add_argument("--out", type=Path, default=REPO / "data/reward_simulation.csv")
    parser.add_argument("--limit", type=int, default=0, help="only this many models")
    args = parser.parse_args()

    paths = sorted(GT_ROOT.glob("*/*.csv"))
    if args.limit:
        paths = paths[: args.limit]

    records, truncated, skipped = [], [], []
    started = time.time()
    for index, path in enumerate(paths, 1):
        database, model = path.parent.name, path.stem
        keys = SORT_KEYS.get(database, {}).get(model, [])
        gt = pd.read_csv(path, dtype=str, keep_default_na=True, low_memory=False)
        if gt.empty or not keys or not scored_columns(gt, keys):
            skipped.append(f"{database}.{model}")
            continue
        full_rows = len(gt)
        if full_rows > args.max_rows:
            gt = gt.iloc[: args.max_rows].copy()
            truncated.append(f"{database}.{model}")

        context = model_context(gt, keys)
        bits = information_content(gt, keys)
        for policy_name, policy in policies.ALL.items():
            pred = policy(gt, keys)
            # Score the submission once; every variant is a different way of
            # reading the same two measurements.
            scored = {
                "plain": cell_f1(gt, pred, keys),
                "weighted": weighted_cell_f1(gt, pred, keys),
            }
            row = {
                "db": database, "model": model, "policy": policy_name,
                "rows": len(gt), "full_rows": full_rows,
                "scored_cols": len(scored_columns(gt, keys)),
                "baseline": context["unweighted_floor"], "bits_per_cell": bits / max(len(gt) * len(gt.columns), 1),
                "official_pass": official_verdict(gt, pred, keys),
            }
            for variant_name, variant in VARIANTS.items():
                row[variant_name] = variant(scored, context)
            records.append(row)
        print(f"[{index}/{len(paths)}] {database}.{model} ({full_rows:,} rows)", flush=True)

    frame = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\nwrote {len(frame):,} rows to {args.out} in {time.time() - started:.0f}s")
    print(f"models scored: {frame['model'].nunique()} | truncated: {len(truncated)} | skipped: {len(skipped)}")
    if truncated:
        print("truncated:", ", ".join(truncated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
