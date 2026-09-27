"""Handing out batches of tasks, indefinitely.

In supervised training an example is a fixed input and target, so a second
pass over it is repetition. A task here is an environment: what comes back
depends on the policy acting in it and on sampling, so revisiting a task after
an update yields genuinely new trajectories. A hundred tasks can therefore
support far more than a hundred gradient steps, and the dataset has to keep
yielding batches rather than running dry partway through a long run.

Kept separate from the training-framework binding so it can be tested without
one.
"""
from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def cyclic_batch(items: Sequence[T], batch_size: int, index: int) -> list[T]:
    """The `index`-th batch, wrapping around the list as often as needed.

    Wrapping is over the item list rather than over whole batches, so when the
    batch size does not divide the list evenly no item is visited less often
    than the rest - a tail that only appeared in short batches would be
    trained on less for no reason.
    """
    if not items or batch_size <= 0:
        return []
    start = (index * batch_size) % len(items)
    return [items[(start + offset) % len(items)] for offset in range(batch_size)]


def batches_per_pass(items: Sequence[T], batch_size: int) -> int:
    """How many batches make one pass over the tasks.

    This is what the trainer reports progress against; a run may ask for more.
    """
    if not items or batch_size <= 0:
        return 0
    return (len(items) + batch_size - 1) // batch_size
