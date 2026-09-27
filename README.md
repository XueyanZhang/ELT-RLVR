# An RL Environment for ELT-Bench

A Tinker RLVR environment for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench):
a language model builds ELT pipelines, the environment executes them against a real
warehouse, and the reward comes from comparing the tables it produced against the
benchmark's ground truth.

**The design, the measurements behind it, and the limitations are in
[`design.md`](design.md).**

## Setup

```bash
git clone --recurse-submodules <this repo> && cd ELT-RLVR
python -m venv .venv && .venv/bin/pip install -e .

# Source tables and ground truth for the small-task pool
.venv/bin/python scripts/fetch_source_data.py --max-rows 10000
```

Training needs [`tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook) and `torch >= 2.10`:

```bash
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
pip install -e ./tinker-cookbook
export TINKER_API_KEY=...
```

## The three checks

### 1. Credential-free local integration test

```bash
.venv/bin/python -m pytest tests/ -q
```

**129 tests, no credentials, no Docker, no network.** Covers the comparator against
real ground-truth columns, alignment over all 73 fixture tables, the reward under
every synthetic submission, and the tools against a live DuckDB.

### 2. A rollout on an official ELT-Bench task

```bash
PYTHONPATH=./tinker-cookbook python scripts/verify_env.py --task cs_semester
```

### 3. One Tinker RL optimization step

```bash
export TINKER_API_KEY=...
PYTHONPATH=./tinker-cookbook python scripts/train_grpo.py \
    tasks=trains,retail_world,computer_student,movie,world,car_retails \
    holdout_fraction=0 group_size=6 groups_per_batch=6 max_steps=1 \
    max_turns=45 max_trajectory_tokens=40960 max_tokens=4096 \
    log_path=/tmp/elt-main
```

This is the run reported in `design.md`: 36 rollouts on `Qwen/Qwen3.5-4B`
(LoRA rank 32), all 36 graded, 22 non-zero, mean reward 0.4134. 

Summarise any run with:

```bash
python scripts/analyze_run.py /tmp/elt-main
```

## Layout

| | |
| --- | --- |
| `elt_rlvr/reward.py` | scoring one data model: cells weighted by information, less what guessing earns |
| `elt_rlvr/grading.py` | scoring a finished rollout, with the benchmark's own verdict alongside |
| `elt_rlvr/tools.py` | the six actions a rollout is made of |
| `elt_rlvr/env.py` | the binding to `tinker-cookbook`: tools, episode, dataset |
| `elt_rlvr/destinations.py` | DuckDB behind the benchmark's own connector interface |
| `elt_rlvr/loading.py` | source CSVs into a warehouse, checked against the expected row counts |
| `elt_rlvr/task.py` | one ELT-Bench task, read from the benchmark's files |
| `elt_rlvr/policies.py` | synthetic submissions used to characterise the reward |
| `elt_rlvr/batching.py` | cycling through the task pool so no task is starved |
| `elt_rlvr/_official.py` | the benchmark's comparator, imported unmodified |
| `scripts/train_grpo.py` | the training entry point |
| `scripts/simulate_reward.py` | 202 models x 11 submissions x 5 candidate reward designs |
| `scripts/analyze_run.py` | per-task rewards, turn usage and stop reasons for a finished run |
| `scripts/fetch_source_data.py` | pulls only the tasks you need out of the 4 GB archive |
| `scripts/verify_env.py`, `verify_group.py`, `verify_grading_survives.py` | end-to-end checks that need no model |

Everything except `env.py` is free of the training framework, which is why the
tests run without `torch`.

`third_party/ELT-Bench` is a submodule pinned to one commit, and its comparator is
imported unmodified.

