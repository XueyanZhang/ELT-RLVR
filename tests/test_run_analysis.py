"""The run analyser must find the grader's output wherever the trainer put it.

Grading moved from the env's per-step reward_fn to the group builder's
compute_group_rewards, which relocated the grader's metrics from the last
step's `metrics` to the rollout's top-level `trajectory_metrics`. Both
layouts exist on disk, so the reader has to handle both.
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "analyze_run", Path(__file__).resolve().parents[1] / "scripts" / "analyze_run.py"
)
analyze_run = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(analyze_run)


def test_reads_group_level_grade_from_trajectory_metrics():
    """Group-graded layout: the grade is top-level, no step carries a reward."""
    rollout = {
        "trajectory_metrics": {"reward": 0.927, "official_srdt": 0.0, "models_attempted": 1.0},
        "steps": [{"metrics": {"stop/completed": 1.0}}],
    }
    assert analyze_run.final_metrics(rollout)["reward"] == 0.927


def test_falls_back_to_the_step_that_carried_the_grade():
    """Legacy layout: trajectory_metrics is empty, the grade is on the last step."""
    rollout = {
        "trajectory_metrics": {},
        "steps": [
            {"metrics": {"stop/tool_stopped": 1.0}},
            {"metrics": {"reward": 0.716, "official_srdt": 0.0}},
        ],
    }
    assert analyze_run.final_metrics(rollout)["reward"] == 0.716


def test_ungraded_rollout_reports_no_reward():
    """A rollout the grader never reached must stay distinguishable from a zero."""
    rollout = {"trajectory_metrics": {}, "steps": [{"metrics": {"stop/max_tokens": 1.0}}]}
    assert "reward" not in analyze_run.final_metrics(rollout)
