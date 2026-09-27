"""Turning a finished rollout into one number.

The reward is a property of the warehouse, not of what the agent said about
it, so grading reads the tables that are actually there and compares them with
the benchmark's ground truth. A model the agent never built, or built with
broken SQL, is simply absent - which scores zero rather than raising.

Both verdicts come back from one pass: the graded reward we train on, and the
benchmark's own pass/fail, reported unchanged so the two stay comparable.
"""
import pytest

from elt_rlvr.destinations import DuckDBConnector
from elt_rlvr.grading import grade_rollout, model_baselines
from elt_rlvr.loading import load_sources
from elt_rlvr.task import load_task

TASK = "cs_semester"

# Every test here reads the benchmark's ground truth from `data/`.
pytestmark = pytest.mark.needs_benchmark_data

# A correct submission for one of the three models. Two subtleties decide it:
# a failing grade is recorded as NULL rather than "F", and "the highest
# satisfaction score for the course" is that course's own maximum, not the
# maximum on the scale - only one of the thirteen courses distinguishes them.
GOOD_COURSES = """
WITH reg AS (
    SELECT r.*, MAX(CAST(r.sat AS INT)) OVER (PARTITION BY r.course_id) AS course_best
    FROM cs_semester.registration r
)
SELECT c.course_id, c.name, c.credit,
       COUNT(*) FILTER (WHERE reg.grade = 'A')                                   AS num_students_got_A,
       100.0 * COUNT(*) FILTER (WHERE CAST(reg.sat AS INT) = reg.course_best) / COUNT(*)
                                                                                 AS per_of_students_highest_satisfaction_score_for_the_course,
       COUNT(*) FILTER (WHERE reg.grade IS NULL)                                 AS number_of_students_fail,
       100.0 * COUNT(*) FILTER (WHERE s.type = 'UG') / COUNT(*)                  AS percentage_of_UG,
       AVG(CAST(reg.sat AS DOUBLE))                                              AS average_satisfying_degree
FROM cs_semester.course c
JOIN reg                   ON reg.course_id = c.course_id
JOIN cs_semester.student s ON s.student_id = reg.student_id
GROUP BY c.course_id, c.name, c.credit
"""


@pytest.fixture(scope="module")
def task():
    return load_task(TASK)


@pytest.fixture(scope="module")
def baselines(task):
    return model_baselines(task)


@pytest.fixture
def warehouse(task):
    connector = DuckDBConnector({})
    load_sources(connector, task.name, task.source_dir)
    yield connector
    connector.close()


def test_an_untouched_warehouse_scores_zero(task, warehouse, baselines):
    grade = grade_rollout(task, warehouse, baselines=baselines)

    assert grade.reward == 0.0
    assert grade.official_srdt == 0.0
    assert all(score.error for score in grade.models)


def test_copying_the_answer_scores_one(task, warehouse, baselines):
    """Grading is checked against its own ceiling: if a table identical to the
    ground truth did not score 1.0, nothing below it would mean anything."""
    for model in task.targets:
        gt = task.ground_truth_dir / f"{model}.csv"
        warehouse.query(
            f"CREATE TABLE {task.name}.{model} AS "
            f"SELECT * FROM read_csv('{gt}', all_varchar = true)"
        )

    grade = grade_rollout(task, warehouse, baselines=baselines)

    assert grade.reward == pytest.approx(1.0)
    assert grade.official_srdt == pytest.approx(1.0)


def test_a_nearly_correct_model_earns_most_of_its_credit_where_the_benchmark_gives_none(
    task, warehouse, baselines
):
    """The case the whole reward design exists for. This submission gets every
    row and all but two columns right; the benchmark scores it exactly as it
    scores an empty warehouse."""
    warehouse.query(f"CREATE TABLE {task.name}.courses AS {GOOD_COURSES}")

    grade = grade_rollout(task, warehouse, baselines=baselines)
    courses = next(m for m in grade.models if m.name == "courses")

    assert courses.official_pass is True
    assert courses.reward > 0.9
    # Two of three models were never attempted, so the task-level reward is
    # the average over all of them.
    assert 0.3 < grade.reward < 0.4


def test_a_model_built_from_nothing_is_scored_not_raised(task, warehouse, baselines):
    warehouse.query(f"CREATE TABLE {task.name}.courses AS SELECT 1 AS course_id")

    grade = grade_rollout(task, warehouse, baselines=baselines)
    courses = next(m for m in grade.models if m.name == "courses")

    assert courses.reward == 0.0


def test_metrics_carry_enough_to_diagnose_a_training_run(task, warehouse, baselines):
    warehouse.query(f"CREATE TABLE {task.name}.courses AS {GOOD_COURSES}")

    metrics = grade_rollout(task, warehouse, baselines=baselines).metrics

    assert metrics["reward"] == pytest.approx(
        grade_rollout(task, warehouse, baselines=baselines).reward
    )
    assert metrics["official_srdt"] > 0
    assert metrics["models_attempted"] == 1.0
    assert "reward/courses" in metrics


def test_baselines_depend_only_on_the_ground_truth(task, baselines):
    """They are precomputed per task and reused across every rollout of it,
    which is only sound because no submission can influence them."""
    assert set(baselines) == set(task.targets)
    assert all(0.0 <= b < 1.0 for b in baselines.values())
    assert model_baselines(task) == baselines
