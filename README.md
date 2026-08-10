# dqn-combat-sim

A hand-written Double-DQN (with Prioritized Experience Replay) trained against a custom
1-vs-1, top-down, discrete-action combat simulation built on Gymnasium. Everything —
environment physics, the replay buffer, the network, the training loop, evaluation, and a
live pygame viewer — is implemented from scratch in this repo (no RL library).

The learning agent (the "soldier") is dropped onto a rectangular map opposite a scripted
"enemy," and has to close distance, track the enemy's bearing, and time its shots against a
ranged, cooldown-gated, accuracy-by-range weapon, while the enemy — which the trainer can ramp
from harmless to fully lethal via a difficulty curriculum — tries to do the same back.

## Repository layout

```
envs/            CombatEnv (the Gymnasium env) and the pure-function physics it's built from
  combat_env.py    reset()/step()/observation — the core simulation, see "The environment" below
  ballistics.py    accuracy/damage-by-range lookup tables, line-of-sight check, shot resolution
  geometry.py      distance, bearing, and continuous-angle <-> discrete-orientation-bin math
  spawner.py       randomized, separation-respecting start positions for reset()

agents/          Policies that can act in CombatEnv, and what DQN trains from
  dqn.py           QNetwork (MLP) + DQNAgent: epsilon-greedy acting, Double DQN target,
                   target-network sync (hard or Polyak), optimizer, checkpoint save/load
  heuristic_agent.py  scripted baseline: face the enemy, hold a standoff range, fire when ready
  random_agent.py     samples the action space uniformly
  replay_buffer.py    ReplayBuffer (uniform) and PrioritizedReplayBuffer (proportional PER)

training/        Entrypoints that drive envs/ + agents/
  train.py            main training loop: curriculum, buffer seeding, checkpointing, eval
  train_multi_seed.py fan one config out across several seeds as parallel OS processes
  evaluate.py          fixed-seed DQN-vs-random-vs-heuristic comparison from a checkpoint
  run_heuristic.py      standalone heuristic-baseline rollout, prints aggregate stats
  watch_policy.py       real-time pygame playback of any policy (dqn/heuristic/random)

logging_utils/   Per-run logging
  run_logger.py        stdlib logging + tensorboard + config snapshot, one call at run start
  async_csv_logger.py  background-thread CSV writer so eval_log.csv never blocks training

rendering/
  pygame_renderer.py   read-only pygame GUI for CombatEnv(render_mode="human")

config/          All tunables live here, not in code (see "Configuration" below)
  sim_default.yaml      env: (map/gun/HP/movement), agent: (heuristic), render: (pygame)
  training_default.yaml reward:, train: (DQN hyperparameters + curriculum), eval:

tests/           pytest unit/integration tests — see "Tests" below
runs/            training output, one folder per run (gitignored)
view_results.py  standalone plotting tool for runs/ eval_log.csv data (see below)
```

`notebooks/` (a scratch/reference notebook, unrelated to the rest of the pipeline) is
gitignored and not covered here.

## Setup

```bash
conda activate 312ml
pip install -r requirements.txt
```

`torch` needs a CUDA-enabled build for training — see the comment in `requirements.txt`
(`pip install torch --index-url https://download.pytorch.org/whl/cu121` or similar for your
CUDA version). `DQNAgent` refuses to run at all if `torch.cuda.is_available()` is false — it
raises immediately rather than silently falling back to CPU, since this project trains on a
CUDA GPU by design.

## The environment

`CombatEnv` (`envs/combat_env.py`) is a `gymnasium.Env` with a flat discrete action space and
a flat continuous (`Box`) observation space — no image/grid input.

**Action space** — one integer encoding three independent choices per tick:
- **move direction**: stay, or one of 8 compass directions (normalized to a fixed step size,
  so diagonals move the same distance per tick as cardinals)
- **fire**: yes/no
- **orientation bin**: which of `n_bins = 360 / bin_size_degrees` discrete headings to face
  this tick (orientation is a direct action, not something that turns gradually)

`action_space = Discrete(9 * 2 * n_bins)` — 180 actions at the default `bin_size_degrees: 36`
(`n_bins = 10`).

**Observation space** — a `Box` of `12 + 2*n_bins` floats (32-dim at the default bin size):
relative position to the enemy (dx, dy); the soldier's own ammo, max weapon range, fire
cooldown remaining, and accuracy-at-current-range; the same four fields mirrored for the
enemy; both units' current HP; and each unit's orientation as a one-hot vector over `n_bins`.

**Line of sight and shot resolution** (`envs/ballistics.py`) — LOS is exact-bin matching: a
shot only has a chance to hit if the shooter's orientation bin equals the target's *true*
bearing bin (`check_los`). If LOS holds, hit probability and damage are both independently
range-interpolated from lookup tables (`gun.accuracy_table` / `gun.damage_table` in
`sim_default.yaml`, `np.interp` under the hood — clamped at the table's ends). Firing consumes
one round of ammo and starts a cooldown regardless of whether the shot has LOS; firing with 0
ammo is a no-op ("empty gun fire") that doesn't start a cooldown.

**The enemy** turns to face the soldier's *true* bearing and fires back under the same
ballistics rules, but its reaction speed, fire rate, and whether it fires at all are all
runtime-tunable via `CombatEnv.set_enemy_difficulty()` — this is the hook `training/train.py`'s
curriculum uses to make the enemy progressively harder over the course of a run (see "Training
curriculum" below) without rebuilding the environment.

**Reward** (`_get_reward` in `combat_env.py`, weights in `training_default.yaml`'s `reward:`)
is a sum of: a flat per-tick step penalty; scaled damage dealt / damage taken; a one-time
penalty the tick ammo hits zero; a small dense reward for facing the enemy's true bearing
every tick (deliberately capped below the step penalty so passively "looking but never
shooting" is still net-negative); a bonus/penalty for firing while facing/not-facing the enemy;
a signed distance-closing term (telescopes to net progress over an episode, so oscillating back
and forth nets ~0 — it can't be farmed); and terminal win/lose/timeout bonuses.

**Episode ends** in `"win"` (enemy HP hits 0 — a simultaneous double-kill also counts as a
win), `"loss"` (soldier HP hits 0), or `"timeout"` (`episode.max_timesteps` reached with both
sides alive).

## Agents

- **`DQNAgent`** (`agents/dqn.py`) — an MLP `QNetwork` (configurable hidden layers/activation)
  trained with Double DQN targets (`agents/dqn.py`'s `train_step`: next action selected by the
  online network, evaluated by the target network) and a choice of hard-sync or Polyak-averaged
  target updates. Supports linear epsilon decay with a curriculum "bump" hook
  (`bump_epsilon` — re-injects exploration on a curriculum stage change without restarting or
  shortening the original decay schedule) and optional linear learning-rate decay applied live
  in `train_step`. `DQNAgent.from_config` builds one from the `train:` section of
  `training_default.yaml` — the same helper `train.py`, `evaluate.py`, and `watch_policy.py`
  all use, so a checkpoint's architecture only has to match its `train_config` once, not per
  script.
- **`HeuristicAgent`** (`agents/heuristic_agent.py`) — always turns to face the enemy's true
  bearing, holds a preferred standoff range (closing in if farther, holding position within
  it), and fires whenever the gun is off cooldown and loaded. Used as an evaluation baseline
  and, optionally, as a source of demonstration transitions during training (see "Replay
  buffer seeding" below).
- **`RandomAgent`** (`agents/random_agent.py`) — samples the action space uniformly; the floor
  baseline for evaluation comparisons.

### Replay buffers

`agents/replay_buffer.py` has two circular (preallocated-numpy, FIFO-on-overflow) buffers:
- **`ReplayBuffer`** — plain uniform sampling.
- **`PrioritizedReplayBuffer`** — proportional-priority PER (Schaul et al.): samples with
  probability `p_i^alpha / sum(p_j^alpha)` instead of uniformly, so rare-but-informative
  transitions (e.g. one of the few wins early in training) get replayed more than their
  share, with importance-sampling weights (`(N * P(i))^-beta`, beta annealed start→end over
  training) to correct the resulting sampling bias. New transitions are pushed at max priority
  so every transition is guaranteed to be sampled at least once before its priority is known.
  This is what `train.py` actually uses by default.

## Training

```bash
conda activate 312ml
python -m training.train --run-name my_run
```

The loop in `training/train.py` is standard off-policy DQN — act, step the env, push the
transition, periodically sample a batch and call `train_step` — but with several optional,
config-gated features layered on top (all controllable from `training_default.yaml`'s `train:`
section, all off or neutral by default so the plain loop is what you get without opting in):

### Training curriculum

A scripted enemy at full difficulty (instant tracking, full-rate return fire) turns out to be
unbeatable from a cold start — see the reasoning captured directly in
`training_default.yaml`'s `train.curriculum` comments, including which configurations were
tried and empirically failed. `train.curriculum.stages` is a list of enemy difficulty settings
(from "can't fire at all" up to full difficulty) that `CombatEnv.set_enemy_difficulty()`
applies live; training advances to the next stage once a *separate* curriculum-check eval's win
rate stays above `win_rate_threshold` for `convergence_window` consecutive checks. The standard
`eval_log.csv` eval always runs at full difficulty regardless of curriculum stage, so it's one
consistent metric across a whole run. Stage advances optionally re-inject epsilon exploration
(`curriculum.epsilon_bump`) since a policy that's gone mostly-greedy against an easier stage has
little incentive to go re-discover what changed.

### Replay-buffer seeding

If `train.replay_buffer.buffer_seeding.enabled`, the buffer is periodically topped up with
fresh `HeuristicAgent` rollouts (matching the *current* curriculum difficulty) so
demonstration-quality "how to aim and engage" transitions don't get evicted out of the
FIFO buffer as self-play accumulates. Off by default.

### Checkpoints, evaluation, and logging

Each run creates `runs/<run-name>_<timestamp>/` containing:
- `config_snapshot.yaml` — the fully-resolved config actually used for this run
- `run.log` — everything logged during the run, including one line per `log_interval_steps`
  window summarizing the episodes completed in it (avg reward/length/win-rate/hit-rate/ammo
  used) — also streamed to stdout, so it's your live terminal readout. There's no per-episode
  CSV; `run.log`/stdout is the record.
- `tensorboard/` — live loss/epsilon/learning-rate/episode/eval curves
  (`tensorboard --logdir runs/<run>/tensorboard`)
- `checkpoints/checkpoint_step<N>.pt` — periodic checkpoints, one every
  `train.checkpoint.interval_steps` env steps (a step count, not an episode count, so it means
  the same thing regardless of episode length)
- `checkpoints/checkpoint_best.pt` — the checkpoint with the best greedy eval win rate seen so
  far in the run
- `eval_log.csv` — one row per in-training evaluation pass (every
  `train.checkpoint.eval_interval_steps`): win/loss/timeout rate, avg reward, avg
  steps-to-kill, avg soldier/enemy HP remaining, avg shots fired, avg damage per shot, avg
  shot accuracy, empty-gun-fire attempts (see `training/evaluate.py:eval_log_row` for the exact
  column list). Written by a background thread (`logging_utils/async_csv_logger.py`) so the
  disk write never blocks the CUDA training loop.
- `final_summary.json` — run name, total steps, episodes completed, and best eval win rate,
  written when training completes.

To override hyperparameters, either edit `config/training_default.yaml` directly or pass
`--train-config path/to/alternate.yaml` (same pattern for `--env-config` / `--reward-config`
/ `--agent-config`).

### On training speed

Headless training has no artificial throttling anywhere — there's no sleep/pacing call in
`training/train.py` (that only exists in `watch_policy.py`, to match GUI playback to real
time). If training still feels slow, it's not that; measured on this machine, the environment
itself steps at ~80k steps/sec standalone, but a full training step — GPU forward pass,
epsilon-greedy action, backward pass, optimizer step — runs at only ~800 steps/sec, because
each of those is a separate small CUDA kernel launch and a small batch (128 samples) is
nowhere near enough work to hide that per-call dispatch overhead. That's inherent to updating
the network on every few environment steps; raising `train.train_every_n_steps` in
`config/training_default.yaml` trades update frequency for wall-clock speed and is the main
lever available for this.

### Training multiple seeds at once

```bash
python -m training.train_multi_seed --seeds 1,2,3,4,5 --run-name my_run
```

Since a single run is CPU-bound (see above) rather than GPU-bound, one run alone only occupies
a slice of both. `train_multi_seed.py` launches one full `train.py` run per seed as a separate
OS process — same config, different `train_config.seed` — so their CPU-heavy env-stepping
loops overlap across cores and their CUDA calls interleave on the GPU, raising utilization
instead of leaving the GPU mostly idle between single-run kernel launches.

One timestamp is picked for the whole batch of seeds (not one per seed), so all of them land
together under one shared experiment folder: `runs/<run-name>_<timestamp>/<seed>/`, each
seed's folder identical in contents to a normal single run. `--seeds` accepts comma-separated
values and/or ranges (`0-4`, `1,5-7,20`, and mixes of both). `--max-parallel` caps concurrency
(default: `cpu_count - 2`) — lower it if the machine is also running something else on the
GPU/CPU. All other flags (`--env-config` / `--reward-config` / `--train-config`) match
`train.py`.

## Evaluating a checkpoint

```bash
python -m training.evaluate --checkpoint runs/<run>/checkpoints/checkpoint_best.pt
```

Rolls out the checkpoint's greedy (epsilon=0) policy against the random and heuristic
baselines on identical fixed seeds (see `training_default.yaml`'s `eval:` section), prints a
comparison table, and saves the full results next to the checkpoint as
`<checkpoint>.eval.json`. Use `--episodes` / `--seed-base` to quickly override the episode
count/seed range without a separate config file. `--train-config` must match whatever
architecture the checkpoint was actually trained with.

## Running the heuristic baseline standalone

```bash
python -m training.run_heuristic
```

Rolls out `HeuristicAgent` for 5000 fixed-seed episodes by default (`--episodes` to change)
under the current env/reward/agent config and prints the aggregate outcome straight to the
terminal — the same metrics `eval_log.csv` tracks (win/loss/timeout rate, avg reward, hit
rate, damage per shot, etc.), but no files are written. Use `evaluate.py --checkpoint` instead
for a saved comparison against a trained DQN checkpoint. `--enemy-fire` / `--no-enemy-fire`
override whether the enemy shoots back (mutually exclusive; default follows
`training_default.yaml`'s `eval.enemy_fire_enabled`).

## Watching a policy play (GUI)

```bash
python -m training.watch_policy --policy dqn --checkpoint runs/<run>/checkpoints/checkpoint_best.pt
python -m training.watch_policy --policy heuristic
python -m training.watch_policy --policy random
```

Opens a pygame window and plays episodes paced to real time (`sim_default.yaml`'s
`env.timestep_duration_s`), by sleeping off whatever's left of each tick's time budget after
`step()` + `render()` — playback only ever slows down to match the budget, never speeds up, so
a slow render frame can't desync the sim. Add `--no-enemy-fire` to disable the scripted
enemy's return fire. `--episodes`/`--seed` control how many episodes to watch and their
starting seed.

## Viewing training results

```bash
python view_results.py                              # scan runs/ and pick from a picker window
python view_results.py <run-name>                    # go straight to a specific run
python view_results.py <run-name> --metric win_rate --ci 0.95 --out plot.png
```

`view_results.py` is a standalone plotting tool (not wired into the `training/` package) for
`eval_log.csv` data. With no argument it scans `runs/` for every folder containing an
`eval_log.csv` — either directly (a single-seed run) or one level down in per-seed
subfolders (a `train_multi_seed.py` experiment) — and opens a tkinter picker window listing
each one with its seed count. Given a run, it plots two stacked charts against
`checkpoint_step`: the chosen `--metric` (default `avg_reward`) on top, and win/loss/timeout
rate on the bottom. For a multi-seed run, both charts show the **mean across seeds** with a
confidence-interval band (never the individual per-seed curves); for a single-seed run, the
line is plotted with no band. `--out` saves a PNG instead of opening an interactive window.

## Configuration

Every tunable value lives in one of two YAML files under `config/`, not in code. Each is
commented in place — read the file itself for what each field does; the tables below are just
a map of what lives where.

| File | Section | Controls | Consumed by |
|---|---|---|---|
| `config/sim_default.yaml` | `env:` | map size, timestep pacing, episode length, spawn rules, movement, gun/ammo, accuracy/damage tables, HP | `CombatEnv` (all entrypoints) |
| | `agent:` | the heuristic baseline's standoff range | `evaluate.py`, `run_heuristic.py`, `watch_policy.py`, `train.py` (buffer seeding) |
| | `render:` | pygame window size, colors, HP bar/orientation-indicator geometry, FPS | `watch_policy.py` (render_mode="human" only) |
| `config/training_default.yaml` | `reward:` | reward shaping weights and terminal bonuses/penalties | `CombatEnv` (all entrypoints) |
| | `train:` | RNG seed, DQN network architecture, optimizer (+ LR decay), replay buffer (+ PER + buffer seeding), target-network sync, epsilon schedule (+ curriculum bump), enemy-difficulty curriculum stages, total step budget (`num_steps_total`), checkpoint/eval-interval step counts | `train.py`, and `evaluate.py`/`watch_policy.py` when loading a checkpoint (must match its architecture) |
| | `eval:` | episode count / seed range / enemy-fire toggle for the standalone `evaluate.py --checkpoint` comparison | `evaluate.py`, `run_heuristic.py` (defaults) |

Every entrypoint accepts `--env-config` / `--reward-config` / `--train-config` /
`--agent-config` / `--render-config` / `--eval-config <path>` flags to override just that
section, each pointing at a standalone file with that section's flat schema (i.e. an
`--env-config` override file is shaped like just the `env:` block, with no `env:` nesting).
Run any script with `--help` to see its full flag list.

## Tests

```bash
pytest tests/
```

Unit tests for the pure physics functions (`test_geometry.py`, `test_ballistics.py`),
integration tests for `CombatEnv.reset()`/`step()` covering episode termination, reward-term
arithmetic, ammo handling, and the enemy reaction-lag curriculum mechanic
(`test_env_reset_step.py`), and unit tests for `DQNAgent`'s epsilon/learning-rate schedules
(`test_dqn_agent.py`) and `PrioritizedReplayBuffer`'s sampling/priority behavior
(`test_replay_buffer.py`).
