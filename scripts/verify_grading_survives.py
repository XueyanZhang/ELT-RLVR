"""Every rollout is graded, however its episode ended.

An episode can finish on a path that skips the environment's own grader - an
output cut off mid-tool-call, a context budget exhausted. When that happened,
a rollout that had built most of the answer was scored by a flat penalty
alone, and within its group it landed *below* one that submitted nothing and
stopped early. The advantage then pointed at giving up.

Grading for the group instead of for the episode removes that. This checks it:
one rollout builds a correct model and is then killed on a path that skips the
episode grader, another does nothing at all, and the first must still outscore
the second.

    python scripts/verify_grading_survives.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tinker-cookbook"))

from elt_rlvr.env import EltEnvGroupBuilder  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
TASK = "cs_semester"
CORRECT_COURSES = """
WITH reg AS (
    SELECT r.*, MAX(CAST(r.sat AS INT)) OVER (PARTITION BY r.course_id) AS course_best
    FROM cs_semester.registration r
)
SELECT c.course_id, c.name, c.credit,
       COUNT(*) FILTER (WHERE reg.grade = 'A') AS num_students_got_A,
       100.0 * COUNT(*) FILTER (WHERE CAST(reg.sat AS INT) = reg.course_best) / COUNT(*)
           AS per_of_students_highest_satisfaction_score_for_the_course,
       COUNT(*) FILTER (WHERE reg.grade IS NULL) AS number_of_students_fail,
       100.0 * COUNT(*) FILTER (WHERE s.type = 'UG') / COUNT(*) AS percentage_of_UG,
       AVG(CAST(reg.sat AS DOUBLE)) AS average_satisfying_degree
FROM cs_semester.course c
JOIN reg ON reg.course_id = c.course_id
JOIN cs_semester.student s ON s.student_id = reg.student_id
GROUP BY c.course_id, c.name, c.credit
"""


async def run() -> int:
    builder = EltEnvGroupBuilder(
        task_name=TASK, model_name=MODEL, renderer_name=None, group_size=2
    )
    envs = await builder.make_envs()
    warehouses = builder._warehouses  # what the group will be graded on

    # Rollout 0 builds a model correctly, then its episode ends on a path that
    # never reaches the environment's grader.
    warehouses[0].query(f"CREATE TABLE {TASK}.courses AS {CORRECT_COURSES}")
    # Rollout 1 does nothing at all and stops.

    # `trajectory_group` is only used for its length here.
    graded = await builder.compute_group_rewards([None, None], envs)
    await builder.cleanup()

    worked, idled = graded[0][0], graded[1][0]
    print(f"built a model then overflowed : reward {worked:.4f}")
    print(f"did nothing and stopped       : reward {idled:.4f}")

    failures = []
    if worked <= idled:
        failures.append("working scored no better than giving up")
    if worked <= 0:
        failures.append("a correct model earned nothing")
    if graded[0][1].get("official_srdt") is None:
        failures.append("the official verdict was not reported")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS - grading survives an episode that skipped the episode grader")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
