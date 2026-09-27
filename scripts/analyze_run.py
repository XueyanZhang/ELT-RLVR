"""Summarise one training run: what each rollout did, and what it earned.

Reads the rollout transcripts a run leaves behind and reports the things a
reader of the design doc would want to check - the spread of rewards per task,
how rollouts spent their turns, why they stopped, and how our graded reward
compares with the benchmark's own verdict on the same submissions.

    python scripts/analyze_run.py /tmp/elt-main
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def load(run_dir: Path) -> tuple[list[dict], dict]:
    summaries = run_dir / "iteration_000000/train_rollout_summaries.jsonl"
    rollouts = [json.loads(line) for line in summaries.read_text().splitlines()]
    metrics = json.loads((run_dir / "metrics.jsonl").read_text().strip().splitlines()[-1])
    return rollouts, metrics


def task_of(rollout: dict) -> str:
    return next(tag for tag in rollout["tags"] if tag != "elt_rlvr")


def final_metrics(rollout: dict) -> dict:
    """The grader's output.

    Since grading moved into the group builder it arrives as trajectory-level
    metrics. Older runs graded inside the episode, so it sat on whichever step
    ended it; those logs are still worth reading, hence the fallback.
    """
    if "reward" in rollout.get("trajectory_metrics", {}):
        return rollout["trajectory_metrics"]
    for step in reversed(rollout["steps"]):
        if "reward" in step.get("metrics", {}):
            return step["metrics"]
    return {}


def tool_calls(rollout: dict) -> Counter:
    counts: Counter = Counter()
    for step in rollout["steps"]:
        for key, value in step.get("logs", {}).items():
            if key.startswith("tool_call_"):
                counts[value.split("(")[0]] += 1
    return counts


def stop_reason(rollout: dict) -> str:
    last = rollout["steps"][-1].get("metrics", {}) if rollout["steps"] else {}
    reasons = [k.split("/", 1)[1] for k in last if k.startswith("stop/")]
    return reasons[0] if reasons else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rollouts, metrics = load(args.run_dir)

    by_task: dict[str, list[dict]] = {}
    for rollout in rollouts:
        by_task.setdefault(task_of(rollout), []).append(rollout)

    print(f"{len(rollouts)} rollouts over {len(by_task)} tasks, "
          f"{metrics['env/all/total_ob_tokens']:,} observation + "
          f"{metrics['env/all/total_ac_tokens']:,} sampled tokens\n")

    print("=" * 78)
    print("REWARD BY TASK")
    print("-" * 78)
    print(f"{'task':<20}{'rewards':<34}{'median':>8}{'best':>8}{'official':>9}")
    for task in sorted(by_task, key=lambda t: -max(
            (final_metrics(r).get("reward", 0.0) for r in by_task[t]), default=0)):
        scored = [final_metrics(r) for r in by_task[task]]
        rewards = [m.get("reward") for m in scored]
        graded = [r for r in rewards if r is not None]
        shown = " ".join("  -  " if r is None else f"{r:.3f}" for r in rewards)
        srdt = [m.get("official_srdt") for m in scored if m.get("official_srdt") is not None]
        print(f"{task:<20}{shown:<34}"
              f"{statistics.median(graded) if graded else 0:>8.3f}"
              f"{max(graded, default=0):>8.3f}"
              f"{(sum(srdt) / len(srdt) if srdt else 0):>9.2f}")

    print("\n" + "=" * 78)
    print("WHERE THE OFFICIAL METRIC AND OURS DISAGREE")
    print("-" * 78)
    disagree = [
        (task_of(r), final_metrics(r))
        for r in rollouts
        if final_metrics(r).get("reward", 0) > 0
        and final_metrics(r).get("official_srdt", 1) == 0
    ]
    print(f"  {len(disagree)} rollouts earned a graded reward while scoring 0 officially")
    for task, m in sorted(disagree, key=lambda x: -x[1]["reward"])[:8]:
        print(f"    {task:<20}graded {m['reward']:.3f}   official 0.00")

    print("\n" + "=" * 78)
    print("HOW ROLLOUTS SPENT THEIR TURNS")
    print("-" * 78)
    print(f"{'task':<20}{'turns':>7}{'read':>6}{'peek':>6}{'sql':>6}{'define':>8}{'run':>6}{'stop':>16}")
    for task in sorted(by_task):
        for rollout in by_task[task]:
            counts = tool_calls(rollout)
            print(f"{task:<20}{len(rollout['steps']):>7}"
                  f"{counts['read_input']:>6}{counts['peek_source']:>6}{counts['run_sql']:>6}"
                  f"{counts['define_model']:>8}{counts['run_transform']:>6}"
                  f"{stop_reason(rollout):>16}")

    print("\n" + "=" * 78)
    print("WHY EPISODES ENDED")
    print("-" * 78)
    for reason, count in Counter(stop_reason(r) for r in rollouts).most_common():
        print(f"  {reason:<20}{count:>3} / {len(rollouts)}")

    graded = [final_metrics(r).get("reward") for r in rollouts]
    graded = [g for g in graded if g is not None]
    print(f"\n  graded {len(graded)}/{len(rollouts)} | "
          f"non-zero {sum(1 for g in graded if g > 0)} | "
          f"mean {statistics.mean(graded):.4f} | median {statistics.median(graded):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
