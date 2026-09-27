"""Binding the ELT environment to the training framework.

Everything that decides what a rollout means - the actions, the warehouse, the
reward - lives in modules that import nothing from the framework and are tested
without one. This file is the seam: it declares those actions as tools the
model can call, assembles them into a multi-turn episode, and hands the result
to GRPO.

The multi-turn behaviour is not implemented here. It comes from the
conversation accumulating: each turn the whole history is re-rendered as the
prompt, the model answers with a tool call, the environment executes it, and
the result is appended as the next observation. `AgentToolMessageEnv` runs that
loop and records the exact tokens sampled at every step, which is what makes
the trajectory trainable.
"""
from __future__ import annotations

from typing import Annotated, Any, Sequence

import chz

from tinker_cookbook import model_info
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import ToolResult, build_agent_tool_env, simple_tool_result, tool

from elt_rlvr.batching import batches_per_pass, cyclic_batch
from elt_rlvr.destinations import DuckDBConnector
from elt_rlvr.grading import grade_rollout, model_baselines
from elt_rlvr.loading import load_sources
from elt_rlvr.task import TaskSpec, available_tasks, load_task
from elt_rlvr.tools import EltTools


class EltToolkit:
    """The agent-facing surface of `EltTools`.

    One instance per rollout, holding that rollout's warehouse. Each method's
    signature and docstring become the tool's schema, so what is written here
    is what the model is told it can do - the only part of the calling protocol
    we control.
    """

    def __init__(self, tools: EltTools):
        self._tools = tools

    @property
    def elt(self) -> EltTools:
        return self._tools

    @tool
    async def read_input(
        self, name: Annotated[str, "File to read, e.g. 'data_model.yaml' or 'schemas/course.csv'"]
    ) -> ToolResult:
        """Read one of the files describing this task."""
        return simple_tool_result(self._tools.read_input(name))

    @tool
    async def peek_source(
        self,
        table: Annotated[str, "Name of a source table already loaded in the warehouse"],
        n: Annotated[int, "How many example rows to show"] = 5,
    ) -> ToolResult:
        """Show a source table's columns, row count and a few example rows."""
        return simple_tool_result(self._tools.peek_source(table, n))

    @tool
    async def run_sql(
        self, query: Annotated[str, "A single read-only SELECT or WITH statement"]
    ) -> ToolResult:
        """Run a read-only query against the warehouse to explore the data.

        This cannot create or modify tables. Use define_model for that.
        """
        return simple_tool_result(self._tools.run_sql(query))

    @tool
    async def define_model(
        self,
        name: Annotated[str, "Which target data model this query builds"],
        sql: Annotated[str, "A SELECT producing exactly that model's columns"],
    ) -> ToolResult:
        """Declare the query that builds one target data model.

        Recorded rather than run; call run_transform to execute. Calling this
        again for the same model replaces the previous definition.
        """
        return simple_tool_result(self._tools.define_model(name, sql))

    @tool
    async def run_transform(self) -> ToolResult:
        """Build every model defined so far and report how each one went."""
        return simple_tool_result(self._tools.run_transform())

    @tool
    async def submit(self) -> ToolResult:
        """Finish. Call this once the data models are built."""
        return simple_tool_result(self._tools.submit(), should_stop=True)

    def as_tools(self) -> list:
        return [
            self.read_input, self.peek_source, self.run_sql,
            self.define_model, self.run_transform, self.submit,
        ]


async def _no_env_reward(messages: list[Message]) -> tuple[float, dict[str, float]]:
    """Grading happens for the group, not for the episode - see below."""
    return 0.0, {}


def _briefing(task: TaskSpec) -> str:
    """The task, stated once. Everything else the agent must go and look at.

    Two things are spelled out because the first run showed rollouts losing
    turns to them: source tables need their schema prefix, and defining a
    model proves nothing until the transform has been run.
    """
    targets = ", ".join(task.targets)
    return (
        f"Build the data models for the `{task.name}` database.\n\n"
        f"The source tables are already loaded and live in the `{task.name}` schema, "
        f"so every reference to one needs that prefix: `{task.name}.<table>`, not "
        f"`<table>`. Your job is the transformation: write one query per target data "
        f"model so its contents match what `data_model.yaml` describes.\n\n"
        f"Target data models: {targets}\n\n"
        "Read data_model.yaml for the columns each model needs and what they mean, and "
        "the files under schemas/ for what the source tables contain. Use peek_source "
        "and run_sql to check your understanding against the data.\n\n"
        "Then, for each model: define_model to record its query, and run_transform to "
        "execute it. **Nothing counts until run_transform has built the tables** - a "
        "definition alone produces nothing. Run it as soon as you have a first attempt, "
        "read whatever errors come back, and refine from there. Call submit when you "
        "are done or out of ideas."
    )


def build_elt_env(
    task: TaskSpec,
    renderer: Renderer,
    baselines: dict[str, float],
    *,
    max_turns: int = 30,
    max_trajectory_tokens: int = 32768,
    max_generation_tokens: int | None = None,
) -> tuple[Env, Any]:
    """Create one episode: a fresh warehouse, the tools over it, and the grader.

    The warehouse is in-memory and belongs to this episode alone, so rollouts
    cannot see each other and a reset is simply not reusing one. Returned
    alongside the environment so the caller can close it.
    """
    warehouse = DuckDBConnector({})
    # The expected row counts double as the list of tables the task declares,
    # which keeps files the archive carries alongside them out of the warehouse.
    load_sources(warehouse, task.name, task.source_dir, expected=task.expected_rows)
    toolkit = EltToolkit(EltTools(warehouse, task.name, dict(task.inputs), list(task.targets)))

    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[t.to_spec() for t in toolkit.as_tools()]
    )
    env = build_agent_tool_env(
        renderer=renderer,
        tools=toolkit.as_tools(),
        initial_messages=prefix + [{"role": "user", "content": _briefing(task)}],
        reward_fn=_no_env_reward,
        max_turns=max_turns,
        max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        # A generation cut off at the token cap leaves an unparsable tool call.
        # Ending the episode there would throw away the warehouse the rollout
        # had already built and score it as if it had done nothing: across the
        # runs in §6 that discarded 35% of rollouts, some after thirty turns of
        # real work. Losing the turn is the right cost; losing the episode is
        # not. §1.2 says an exhausted budget is scored on the warehouse as it
        # stands, and this is that case.
        terminate_on_parse_error=False,
    )
    return env, warehouse


@chz.chz
class EltEnvGroupBuilder(EnvGroupBuilder):
    """One task, sampled `group_size` times - the unit GRPO centres over.

    Holds only names and paths. The warehouses and the renderer are built in
    `make_envs`, because a builder is pickled and shipped to remote workers and
    a live database connection cannot travel.
    """

    task_name: str
    model_name: str
    renderer_name: str | None
    group_size: int
    max_turns: int = 30
    max_trajectory_tokens: int = 32768
    max_generation_tokens: int | None = None

    async def make_envs(self) -> Sequence[Env]:
        task = load_task(self.task_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, get_tokenizer(self.model_name))
        baselines = model_baselines(task)  # per task, computed once and cached

        envs, warehouses = [], []
        for _ in range(self.group_size):
            env, warehouse = build_elt_env(
                task, renderer, baselines,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
            )
            envs.append(env)
            warehouses.append(warehouse)
        object.__setattr__(self, "_warehouses", warehouses)
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list, env_group: Sequence[Env]
    ) -> list[tuple[float, dict[str, float]]]:
        """Grade every rollout's warehouse, however its episode ended.

        The environment could grade at the end of an episode instead, and did.
        But an episode can end on a path that skips the grader - an output cut
        off at the token cap, a context budget exhausted - and then a rollout
        that built most of the answer is scored only by the flat penalty for
        ending that way. In one run that happened to 11 of 36 rollouts, some
        after forty turns of work.

        The consequence is worse than losing the reward. Within a group, a
        rollout that explored hard and overflowed scored *below* one that
        submitted nothing and stopped early, so the advantage pointed at giving
        up: the environment was training the policy away from doing the task.

        Grading here removes the whole class of problem rather than the two
        paths we happened to find. This method runs once per group after every
        rollout finishes, whatever ended it, and the warehouses are still ours.
        """
        task = load_task(self.task_name)
        baselines = model_baselines(task)
        warehouses = getattr(self, "_warehouses", [])

        rewards: list[tuple[float, dict[str, float]]] = []
        for index in range(len(trajectory_group)):
            if index >= len(warehouses):
                rewards.append((0.0, {"grading_failed": 1.0}))
                continue
            try:
                grade = grade_rollout(task, warehouses[index], baselines=baselines)
                rewards.append((grade.reward, grade.metrics))
            except Exception:
                # A rollout that cannot be graded is data, not a crash: raising
                # here would discard the whole group.
                rewards.append((0.0, {"grading_failed": 1.0}))
        return rewards

    async def cleanup(self) -> None:
        for warehouse in getattr(self, "_warehouses", []):
            try:
                warehouse.close()
            except Exception:
                pass
        object.__setattr__(self, "_warehouses", [])

    def logging_tags(self) -> list[str]:
        return ["elt_rlvr", self.task_name]


class EltDataset(RLDataset):
    """Batches of tasks. One task is one group; one batch is several tasks.

    Batches keep coming past the end of the task list. A task is an
    environment rather than a fixed example, so revisiting it after an update
    produces new trajectories - a hundred tasks support far more than a
    hundred steps, and a run that asked for more would otherwise stop early
    with nothing to say why.
    """

    def __init__(self, builders: list[EltEnvGroupBuilder], batch_size: int):
        self._builders = builders
        self._batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return cyclic_batch(self._builders, self._batch_size, index)

    def __len__(self) -> int:
        """One pass over the tasks, which is what the trainer shows progress
        against. It may be asked for more batches than this."""
        return batches_per_pass(self._builders, self._batch_size)


@chz.chz
class EltDatasetBuilder(RLDatasetBuilder):
    """Splits the available tasks into a training set and a held-out set.

    The split is over tasks, not rows: a policy that had seen a task's data
    models during training would tell us nothing about a new one. Fivetran and
    Bird tasks are interleaved so both land on each side, since selecting tasks
    by size alone skews heavily towards Fivetran.
    """

    model_name: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 30
    max_trajectory_tokens: int = 32768
    max_generation_tokens: int | None = None
    tasks: str = ""          # comma-separated; defaults to everything available
    holdout_fraction: float = 0.25
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        import random

        names = (
            [t.strip() for t in self.tasks.split(",") if t.strip()]
            if self.tasks
            else available_tasks()
        )
        random.Random(self.seed).shuffle(names)
        n_holdout = int(len(names) * self.holdout_fraction)
        holdout, train = names[:n_holdout], names[n_holdout:]

        def builders(subset: list[str]) -> list[EltEnvGroupBuilder]:
            return [
                EltEnvGroupBuilder(
                    task_name=name,
                    model_name=self.model_name,
                    renderer_name=self.renderer_name,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                )
                for name in subset
            ]

        train_dataset = EltDataset(builders(train), self.batch_size)
        holdout_dataset = EltDataset(builders(holdout), self.batch_size) if holdout else None
        return train_dataset, holdout_dataset
