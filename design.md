# ELT-Bench RL

# Summary

In this task, I build a **Tinker RLVR env** and **GRPO recipe** for **ELT-Bench**.
1. Agent operates a data warehouse via **defined tools**
2. Env executes the SQL
3. Reward is computed by contrasting with the ground truth data models.

The core issue is **GRPO reward design**. 
The official SRDT metric is not suitable as an RLVR reward. SRDT is near-zero almost everywhere. GRPO requires rollouts has different advantages. 
I designed a reward with partial credits, making the outcome reward continuous..

Scope is limited to the "T (transformation)" in the ELT task. The "EL (extract/load)" step is pre-loaded by Env.  I learned that T (transformation) is the more challenging task in ELT-bench (SRDEL 57%, SRDT 3.9% only)

**Result** on untrained `Qwen/Qwen3.5-4B`, 6 official tasks × 6 rollouts, one GRPO step
- graded 36/36 (non-zero 22/36)
- all 6 groups carried reward variance

(Also record some of my failures.)

# Reward design

SRDT is a nice metric but it may appear too coarse for GRPO.
`advantage_i = r_i − mean(r_group)`. If a group of rollouts has the same reward, there is no advantage (aka, there is nothing to learn).

I mainly explored the outcome reward, breaking the discrete reward into continuous ones. 
- The reward is computed on the final state of the warehouse. (not process reward)
- The reward assesses the "how far/close the result is from the ground truth"
- The granularity is per cell.

**Interesting finding** (reward hacking oppotunity)
On 202 official models, the data are sparse and mostly repeated values.
`app_store.apps` has 9659 rows, and 91% cells are either 0 or NaN. (as most app has no comments)

**Formula**
step 1. Rare value has more weights
`w(r,c) = -log2(Frequency_of_this_value_in_the_column)`
- Value 0 appears in 8 out of 10 cells, `w = -log2(0.8) = 0.32` (less important)
- Value 5 appears 2 times, `w = -log2(0.2) = 2.32` (more important to get this one correct)

step 2.
`score = (Weight_sum_of_correct_cell / Weight_sum) x (2|gt|/(|gt|+|t|))`
- left term: the share of the answer's information the submission recovered
- right term: equal to 1 when #rows matches the ground truth (gt)
    - `|gt|` - #row of ground truth
    - `|t|` - #row of the outcome by agent

step 3. offset the constant filling 
`reward = max( 0, (score - baseline) / (1 - baseline) )`
- baseline score is computed by filling with most common values in that table
    - basically constant filling scores 0
    - correct answer is 1

**Examples**
|  cases   |  reward   |
| --- | --- |
| a correct submission |   1.0   |
| a constant filler, at any granularity |   0.0  |
| half the rows, done right |  0.62   |

# Env design

ELT-Bench stats: 100 pipelines, 835 sources, 203 target data model
Agent observes config.yaml, data_model.yaml, schemas/, Airbyte

Env is stateful. The state of the warehouse is updated by agent actions. Every step depends on prior steps, aka sequential decisions.
Each rollout must start from a clean state (reset); otherwise expisodes are contaminated.

```
Inner loop (per turn, ~30 times per rollout)
step 1. Tinker samples the LLM response -> pass tokens to Env
step 2. Env extracts the tool call -> executes -> result appended to conversation
step 3. go to step 1, now with a longer prompt

Outer loop (once per training step)
step 1. all 36 rollouts finish -> each graded on its final warehouse state
step 2. advantages centred within each group (one task = one group)
step 3. Tinker forward_backward
```

Action space (tools defined)
- read task files
- sample a source table
- run a read-only query
- define a model's SQL
- run the transform
- submit

Termination condition
- agent call `submit`
- max turn
- single generation token limit
- context window limit
- other: verbose, no tool call

The warehouse is rewarded by the final state (regardless of the terminal condition).
- reward=+0.000   advantage=-0.095    12 turns, built nothing, stopped early
- reward=-0.100   advantage=-0.195    35 turns, iterated the whole way

The destination is pluggable as per request. `elt_rlvr/destinations.py (88 lines)`

# Experiment

- Qwen/Qwen3.5-4B
- 6 official ELT-Bench task
- 6 rollouts
- max_turn = 45
- grpo_step = 1

| task | rewards (high to low) | median | official SRDT |
| --- | --- | --- | --- |
| computer_student | 1.000 0.993 0.918 0.912 0.907 0.000 | 0.915 | 0.17 |
| trains | 1.000 1.000 0.954 0.604 0.465 0.465 | 0.779 | 0.33 |
| world | 0.927 0.738 0.648 0.000 0.000 0.000 | 0.324 | 0.00 |
| movie | 0.924 0.398 0.161 0.000 0.000 0.000 | 0.080 | 0.00 |
| retail_world | 0.687 0.618 0.000 0.000 0.000 0.000 | 0.000 | 0.00 |
| car_retails | 0.198 0.184 0.180 0.000 0.000 0.000 | 0.090 | 0.00 |

graded 36/36
non-zero 22/36
mean 0.4134
median 0.2979

**Official metric vs ours**

Applying GRPO's discard rule to the table above:

| | groups with gradient | groups discarded |
| --- | --- | --- |
| official SRDT | 2 / 6 | **4 / 6** (`world`, `movie`, `retail_world`, `car_retails` - uniformly zero) |
| our reward | **6 / 6** | 0 |

- the official metric would leave **2/3 of the sampling producing no gradient**.
- **19 rollouts earned a graded reward while scoring 0 officially**.

Gradient step
- train_step 76.4s
- lr 1e-5
- entropy 0.345 

Agent status
- tool_stopped 16 / 36
- max_turns 12 / 36
- max_tokens 1 / 36
- other 7 / 36 (forgot submission, no tool calling, etc)

**Findings & Limitations**
- Small models like qwen3.5 4b can still solve some of tasks like `computer_student` and `trains`. It struggles with complex task like `car_retails` (which requires three models).
- This step shows that the proposed reward function can generate signals for potential GRPO training; however, whether it keep moving over multi-round is unanswered.
- Only a small subset of the official tasks are selected for this report. The numbers here could be biased. 

# FAQ

## Q1. Where is SRDEL?
To simplify the question, SRDEL is constant 1.0 because it is handled by the Env.
From the paper, the SRDT is the more challenging task.

To add EL steps to agent, we need to add two more actions to `write Airbyte config` and `run the load`.  If we decided to train these two stage separately, SRDEL could be a good RLVR reward.

## Q2. Reward hacking?

**Ground truth leaking**
Ground truth is in CSV file. Our tool `read_input` has a white list. Agent cannot bypass to read any files.

**Bypass transform step** and directly fill the result model
Currently the proposed reward design cannot really distinguish if the warehouse state is reached by transform tool call or direct filling.

**Row order is irrelevant**
The reward is computed by unique key rather than fixed position.

**Abuse process reward**
By design, there is no process reward. The reward is only computed by the final warehouse state. An agent cannot just chat and gain process reward repatedly, like correct SQL grammar.

**Abuse #rows**
Repeated rows or excessive rows are penalized by `2|gt|/(|gt|+|t|)` and each correct row is only match once.

## Q3. What about efficiency?

Sampling dominates in the time taken. Parallel sampling will optimize the efficiency. To achive parallelism, all sources must be read-only and each rollout has its isolated warehouse.

## Q4. Why not using the agent runner?

The builtin agents are SWE-agent, Claude code, and Codex. 
First, we cannot train commercial models like Claude code and Codex.
Second, to train a model, we need to control the toenization process for calculating the step gradients.
Thus, we adopt Qwen3.5 with our own tools.

## Q5. Reproduce the expriment

Please refer to the github repo readmd :)
- Credential-free local integration test, `pytest tests/ -q`
- Credentialed rollout on an official task, `python scripts/verify_env.py --task cs_semester`
- One Tinker RL optimization step, `python scripts/train_grpo.py ... max_steps=1`

## AI Assistance Disclosure
I used Claude and Claude code during the process to explain ideas and write codes. 