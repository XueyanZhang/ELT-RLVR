"""A task is not consumed by being used.

In supervised training an example is a fixed input and target, so a second
pass over it is repetition. A task here is an environment: the policy acts in
it, and what comes back depends on the policy and on sampling. Revisiting a
task after an update produces genuinely new trajectories, which is why 100
tasks can support far more than 100 gradient steps.

The dataset therefore has to keep yielding batches past the end of its task
list, rather than running dry - a training run that asked for more steps than
tasks would otherwise stop early with no explanation.
"""
import pytest

from elt_rlvr.batching import batches_per_pass, cyclic_batch


TASKS = [f"task_{i}" for i in range(5)]


@pytest.fixture
def batch():
    return lambda index: cyclic_batch(TASKS, 2, index)


def test_a_batch_is_as_wide_as_requested(batch):
    assert batch(0) == ["task_0", "task_1"]
    assert batch(1) == ["task_2", "task_3"]


def test_batches_keep_coming_after_the_tasks_run_out(batch):
    """Asking for more steps than there are tasks is normal, not an error."""
    for index in range(20):
        assert len(batch(index)) == 2, f"batch {index} came back short"


def test_every_task_is_visited_at_the_same_rate(batch):
    """Five tasks do not divide into batches of two, so wrapping over whole
    batches would show the tail less often. Wrapping over the task list keeps
    every task's exposure within one visit of every other's."""
    seen = [task for index in range(10) for task in batch(index)]
    counts = [seen.count(task) for task in TASKS]

    assert len(set(seen)) == 5
    assert max(counts) - min(counts) <= 1


def test_one_pass_is_what_length_reports(batch):
    """The trainer shows progress against this; it should mean one pass over
    the tasks even though the dataset can be asked for more."""
    assert batches_per_pass(TASKS, 2) == 3
    assert batches_per_pass([], 2) == 0
