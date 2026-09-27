"""Drive a full episode without a model, and without spending anything.

Everything in a rollout except the sampling is ours: rendering the tools into
a prompt, parsing a tool call back out, executing it against a real warehouse,
feeding the result in as the next observation, and grading what the episode
left behind. This script exercises all of it by replacing the policy with a
scripted sequence of tool calls, so the machinery can be trusted before a
single token is paid for.

    python scripts/verify_env.py --task cs_semester
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tinker-cookbook"))

from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from elt_rlvr.env import build_elt_env  # noqa: E402
from elt_rlvr.grading import model_baselines  # noqa: E402
from elt_rlvr.task import load_task  # noqa: E402

# A competent submission for one of cs_semester's three models. Two subtleties
# decide it: a failing grade is NULL rather than "F", and "the highest
# satisfaction score for the course" is that course's own maximum.
COURSES_SQL = """
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

#: Each turn is a tool call and something its reply must contain, so the run
#: fails loudly if a tool stops doing what the design says it does.
SCRIPTED_TURNS = [
    ("read_input", {"name": "data_model.yaml"}, "students"),
    ("peek_source", {"table": "registration", "n": 3}, "101 rows"),
    ("run_sql", {"query": "SELECT COUNT(*) AS n FROM cs_semester.registration"}, "101"),
    ("run_sql", {"query": "DROP TABLE cs_semester.course"}, "read-only"),
    ("define_model", {"name": "courses", "sql": COURSES_SQL}, "Recorded"),
    ("define_model", {"name": "not_a_target", "sql": "SELECT 1"}, "not a target"),
    ("run_transform", {}, "courses: ok"),
    ("submit", {}, "Submitted"),
]


MODEL_FOR_CHECK = "Qwen/Qwen3.5-4B"


async def run(task_name: str, model_name: str) -> int:
    global MODEL_FOR_CHECK
    MODEL_FOR_CHECK = model_name
    task = load_task(task_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, get_tokenizer(model_name))
    print(f"task={task.name}  model={model_name}  renderer={renderer_name}")
    print(f"targets={list(task.targets)}\n")

    env, warehouse = build_elt_env(task, renderer, model_baselines(task))

    observation, _stop = await env.initial_observation()
    prompt_tokens = len(observation.to_ints())
    print(f"[prompt] {prompt_tokens} tokens; the tool schemas and the briefing")

    from tinker_cookbook.renderers import get_text_content

    message_env = env.message_env  # the tool loop underneath
    result = None
    failures: list[str] = []
    for index, (name, arguments, expected) in enumerate(SCRIPTED_TURNS, 1):
        result = await message_env.step(_sampled_message(renderer, name, arguments))
        # `next_messages` is the whole conversation; the tool's answer is the tail.
        reply = get_text_content(result.next_messages[-1]) if result.next_messages else ""
        ok = expected.lower() in reply.lower()
        if not ok:
            failures.append(f"{name}: expected {expected!r} in reply")
        print(f"[{index}] {'ok ' if ok else 'BAD'} {name}({_short(arguments)})")
        print(f"        {reply.strip().splitlines()[0][:120] if reply.strip() else '(no reply)'}")
        if result.episode_done:
            print(f"\nepisode ended after {index} turns")
            break

    assert result is not None

    # The episode itself returns no reward: grading is a group-level operation
    # so that a rollout is scored however its episode ended. Here we grade the
    # warehouse this episode left behind, which is what the group would do.
    from elt_rlvr.grading import grade_rollout

    grade = grade_rollout(task, warehouse, baselines=model_baselines(task))
    print(f"\nreward  = {grade.reward:.4f}   official SRDT = {grade.official_srdt:.2f}")
    for score in grade.models:
        print(f"  {score.name:<14}{score.reward:.4f}"
              + (f"   ({score.error})" if score.error else ""))
    warehouse.close()

    # One of three models was built, and built correctly.
    if not result.episode_done:
        failures.append("the episode did not end")
    if abs(grade.reward - 1 / 3) > 0.01:
        failures.append(f"expected a reward of 1/3, got {grade.reward:.4f}")
    courses = next(s for s in grade.models if s.name == "courses")
    if abs(courses.reward - 1.0) > 1e-9:
        failures.append("the correct model did not score 1.0")
    if any(s.reward != 0.0 for s in grade.models if s.name != "courses"):
        failures.append("a model that was never built scored above zero")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if not await check_truncation_does_not_discard_the_episode(task.name, MODEL_FOR_CHECK):
        return 1
    print("\nPASS - tools dispatched, read-only enforced, warehouse graded")
    return 0


def _sampled_message(renderer, name: str, arguments: dict) -> dict:
    """Build the message the way a real turn builds it.

    Rather than constructing the parsed object directly, this emits the exact
    text a model would produce, tokenises it, and hands it to the renderer -
    so the tool declarations, the wire format and the parser are all exercised
    together. If they disagree, this is where it shows.
    """
    import json

    text = "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False)
    tokens += renderer.get_stop_sequences()
    message, termination = renderer.parse_response(tokens)
    if not message.get("tool_calls"):
        raise AssertionError(
            f"the renderer did not parse a call to {name!r} "
            f"(termination={termination}, message={message})"
        )
    return message


def _short(arguments: dict) -> str:  # noqa: D401
    return ", ".join(f"{k}={str(v)[:40]!r}" for k, v in arguments.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="cs_semester")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    return asyncio.run(run(args.task, args.model))


async def check_truncation_does_not_discard_the_episode(task_name: str, model_name: str) -> bool:
    """A cut-off generation must cost a turn, not the rollout.

    The model's output is capped, so a long query can end mid-token; what
    arrives is an unparsable tool call. If that ended the episode, everything
    the rollout had already built would go ungraded - which is how 35% of the
    rollouts behind §6 were scored as if they had done nothing.
    """
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook import model_info
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    task = load_task(task_name)
    renderer = get_renderer(model_info.get_recommended_renderer_name(model_name),
                            get_tokenizer(model_name))
    env, warehouse = build_elt_env(task, renderer, model_baselines(task))
    await env.initial_observation()

    # Build the real thing first, then send a truncated call.
    await env.message_env.step(
        _sampled_message(renderer, "define_model", {"name": task.targets[0], "sql": "SELECT 1 AS x"})
    )
    truncated = renderer.parse_response(
        renderer.tokenizer.encode('<tool_call>\n{"name": "run_sql", "argum',
                                  add_special_tokens=False)
        + renderer.get_stop_sequences()
    )[0]
    result = await env.message_env.step(truncated)
    warehouse.close()

    if result.episode_done:
        print("FAIL: a truncated generation ended the episode")
        return False
    print("ok  a truncated generation cost a turn, not the episode")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
