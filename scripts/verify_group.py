"""Check the GRPO plumbing without sampling anything.

A group is the unit the algorithm centres rewards over, so it has to be built
correctly before any of the training matters: the right number of independent
environments, each with its own warehouse, from a builder that survives being
pickled and sent to a worker.

    python scripts/verify_group.py
"""
from __future__ import annotations

import asyncio
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tinker-cookbook"))

from elt_rlvr.env import EltDatasetBuilder, EltEnvGroupBuilder  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
GROUP_SIZE = 4


async def run() -> int:
    failures: list[str] = []

    builder = EltDatasetBuilder(
        model_name=MODEL, batch_size=2, group_size=GROUP_SIZE, seed=0, holdout_fraction=0.25
    )
    train, holdout = await builder()
    train_tasks = {b.task_name for batch in range(len(train)) for b in train.get_batch(batch)}
    holdout_tasks = (
        {b.task_name for batch in range(len(holdout)) for b in holdout.get_batch(batch)}
        if holdout
        else set()
    )
    print(f"train {len(train_tasks)} tasks in {len(train)} batches; holdout {len(holdout_tasks)}")

    if train_tasks & holdout_tasks:
        failures.append(f"a task is in both splits: {sorted(train_tasks & holdout_tasks)}")
    if not holdout_tasks:
        failures.append("no held-out tasks")

    group: EltEnvGroupBuilder = train.get_batch(0)[0]
    print(f"group builder: task={group.task_name} size={group.group_size}")

    # Builders are shipped to remote workers, so they must pickle - which is
    # why a live connection is never a field.
    try:
        revived = pickle.loads(pickle.dumps(group))
        if revived.task_name != group.task_name:
            failures.append("pickling changed the builder")
    except Exception as error:
        failures.append(f"the builder does not pickle: {error}")

    envs = await group.make_envs()
    print(f"made {len(envs)} environments")
    if len(envs) != GROUP_SIZE:
        failures.append(f"expected {GROUP_SIZE} environments, got {len(envs)}")

    # Each rollout must own its warehouse: a table built in one must be
    # invisible to the rest, or rewards stop being a function of one rollout.
    warehouses = getattr(group, "_warehouses", [])
    if len(set(map(id, warehouses))) != len(envs):
        failures.append("environments are sharing a warehouse")
    else:
        warehouses[0].query("CREATE TABLE cs_semester_probe AS SELECT 1 AS x")
        leaked = [
            index
            for index, w in enumerate(warehouses[1:], start=1)
            if "cs_semester_probe"
            in set(w.query("SELECT table_name FROM information_schema.tables")["table_name"])
        ]
        if leaked:
            failures.append(f"a table leaked into environments {leaked}")
        else:
            print("warehouses are isolated: a table in one is invisible to the others")

    prompts = [len((await env.initial_observation())[0].to_ints()) for env in envs]
    print(f"prompt lengths: {prompts}")
    if len(set(prompts)) != 1:
        failures.append("environments in a group started from different prompts")

    await group.cleanup()
    print("cleanup released the warehouses")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS - groups build, isolate and pickle")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
