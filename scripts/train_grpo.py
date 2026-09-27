"""Train a policy on ELT-Bench with GRPO.

A thin front end over `tinker_cookbook.rl.train`: everything specific to this
problem lives in `elt_rlvr`, and everything about the algorithm comes from the
cookbook unchanged. One task forms one group, so the advantage each rollout
receives is measured against its siblings on the same task.

    python scripts/train_grpo.py                      # every available task
    python scripts/train_grpo.py tasks=cs_semester max_steps=1
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tinker-cookbook"))

import chz  # noqa: E402

from tinker_cookbook import cli_utils  # noqa: E402
from tinker_cookbook.rl.train import Config, main  # noqa: E402

from elt_rlvr.env import EltDatasetBuilder  # noqa: E402


@chz.chz
class CLIConfig:
    # Start small: the first run is about whether the loop closes, not about
    # how far the policy gets.
    # Tinker serves a fixed model list; Qwen3.5-4B is the smallest on it.
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str | None = None
    lora_rank: int = 32

    tasks: str = ""            # comma-separated; defaults to every fetched task
    holdout_fraction: float = 0.25
    seed: int = 0

    group_size: int = 8        # rollouts per task - what GRPO centres over
    groups_per_batch: int = 2
    learning_rate: float = 1e-5
    max_turns: int = 30
    max_trajectory_tokens: int = 32768   # hard ceiling on one rollout's context
    max_tokens: int = 4096
    max_steps: int | None = None

    log_path: str | None = None
    wandb_project: str | None = None
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(config: CLIConfig) -> None:
    run_name = (
        f"elt-rlvr-{config.model_name.replace('/', '-')}-"
        f"{config.group_size}group-seed{config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    dataset_builder = EltDatasetBuilder(
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        batch_size=config.groups_per_batch,
        group_size=config.group_size,
        max_turns=config.max_turns,
        max_trajectory_tokens=config.max_trajectory_tokens,
        max_generation_tokens=config.max_tokens,
        tasks=config.tasks,
        holdout_fraction=config.holdout_fraction,
        seed=config.seed,
    )
    log_path = config.log_path or f"/tmp/elt-rlvr/{run_name}"
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
    await main(
        Config(
            learning_rate=config.learning_rate,
            dataset_builder=dataset_builder,
            model_name=config.model_name,
            recipe_name="elt_rlvr",
            renderer_name=config.renderer_name,
            lora_rank=config.lora_rank,
            max_tokens=config.max_tokens,
            wandb_project=config.wandb_project,
            wandb_name=run_name,
            log_path=log_path,
            base_url=config.base_url,
            max_steps=config.max_steps,
        )
    )


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
