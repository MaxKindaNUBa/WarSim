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
  combat_env.py    reset()/step()/observation — the core simulation, see "The simulation
                   environment, in plain terms" below
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

`environment_design_spec.md` — the original design spec the simulation was built from (soldier/
enemy behavior, ballistics, action/observation space, reward — in narrative form). The
"simulation environment" section below is the practical, up-to-date version of the same
material, including a few places the implementation moved past that spec.

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

## The simulation environment, in plain terms

### The premise

Every episode is a 1-on-1 shootout on a flat, featureless rectangular map — no obstacles, no
cover, no terrain. Two combatants: a **soldier**, who is the one thing being trained (every
action it takes is chosen by the DQN policy), and an **enemy**, a fixed, non-learning scripted
opponent. The soldier has to close to an effective range, keep its weapon aimed at the enemy,
and land shots before it runs out of time, health, or ammo — while the enemy does the same back
under a scripted (not learned) targeting routine. There's no story or visual complexity beyond
that; the whole "game" reduces to positions, facing direction, health, and ammo, plus math
derived from those.

An episode ends the instant one side's HP hits zero, or after a fixed number of ticks with
both still alive (see "How an episode ends" below).

### The world and the passage of time

The map is a flat plane with a fixed width/height (`env.map.width` / `height` in
`sim_default.yaml`, default 100x100 map units) — bounded only so movement/rendering stay
cheap and predictable, not because the boundary is part of the tactical problem (everything
either combatant observes is relative distance/bearing, never absolute map position). Neither
combatant can walk outside it.

Time moves in fixed discrete **ticks**, not continuous physics: each `step()` call is one tick
— one movement update, one orientation update, and up to one fire attempt per side, all
resolved together. Each tick nominally represents `env.timestep_duration_s` seconds of
simulated time (default 0.5s), but that only matters for the GUI viewer
(`training/watch_policy.py`), which sleeps between ticks to match real time. Headless training
and evaluation runs have **no throttling at all** — they tick forward as fast as the CPU can
compute, completely decoupled from `timestep_duration_s`.

### The soldier (the learning agent)

The soldier is the only entity anything is "learning" for — everything else in the simulation
exists to give it something to overcome. Each tick, the DQN policy picks one action that bundles
three independent decisions together (see "Action space" below): where to move, which way to
face, and whether to fire.

- **Movement**: one of 8 compass directions, or stand still, each tick. A chosen direction
  always covers the same fixed distance (`env.movement.step_size`, default `2.0` map units) —
  no momentum, no acceleration, and diagonal moves aren't any faster than cardinal ones (the
  direction vector is normalized before being applied). Movement is clipped at the map edges.
  Crucially, **movement and aiming are decoupled** — the soldier can walk northeast while
  facing west; picking a movement direction never changes which way its weapon points.
- **Orientation (aiming)**: a separate choice each tick, from a fixed number of discrete
  "orientation bins" that divide the full 360° circle into equal wedges
  (`env.bin_size_degrees`, default `36°` per bin → 10 bins). Wherever the soldier is currently
  oriented is where it will fire if it chooses to fire that tick — there's no separate aim vs.
  facing concept, and no gradual turning; a chosen orientation takes effect immediately.
- **The gun**: fixed ammo per episode (`env.gun.max_ammo`, default `20`) with no reload — once
  it's empty, firing is still a *legal* action, it just can't do anything. A max effective
  range (`env.gun.max_range`) beyond which shots can never land. A cooldown
  (`env.gun.fire_cooldown_steps`) that must fully count down between shots. And two independent
  range-dependent lookup curves — hit probability (`accuracy_table`) and damage-per-hit
  (`damage_table`) — both of which get *worse* the farther away the target is (see "Ballistics"
  below for exactly how they're used). These same mechanics apply symmetrically to the enemy's
  weapon too.
- **Firing is always legal, and never free.** Attempting to fire — even at zero ammo, even
  facing the wrong way, even beyond max range — always consumes a round and starts the
  cooldown if ammo is available; a "wasted" shot costs the same resource as an aimed one.
  This is deliberate: it's what stops a trained policy from just spamming fire every tick
  regardless of whether it's actually lined up, and it's why `evaluate.py` tracks
  "empty gun fire attempts" as its own metric.
- **HP**: depletes when the enemy lands a hit, by however much that hit's damage roll came out
  to. Reaching 0 ends the episode as a loss.

### The enemy (the scripted opponent)

The enemy is not learning anything — its behavior is a fixed script, deliberately kept simple
so the challenge in this environment comes from range/ammo/timing management against a
*predictable* threat, not from an adaptive one.

- **It never moves.** Its position is wherever it randomly spawned, for the whole episode.
- **Its aim tracks the soldier.** Every tick it recomputes the true compass bearing from itself
  to the soldier's *current* position and snaps its orientation to the nearest bin toward that
  bearing. At full difficulty this is instant and perfect — no turning delay — which is why the
  spawn point, not evasive maneuvering, is the main lever the soldier has over whether the
  enemy is aimed at it.
- **It fires automatically whenever it legally can** — soldier in range, cooldown expired — it
  never "chooses" not to shoot the way a more sophisticated opponent might.
- **Same weapon model as the soldier**: its own ammo/range/cooldown/accuracy/damage curves,
  configured independently (`env.enemy` / the same `gun:` block — soldier and enemy currently
  share one weapon config, but the ballistics code treats them symmetrically either way).
- **Spawn**: randomized every episode, subject to a minimum-separation constraint from the
  soldier's own randomized spawn point (`env.episode.min_spawn_separation`) so episodes never
  start at point-blank range purely by chance.

**Difficulty is tunable at runtime**, beyond what the original design spec called for: reaction
speed, fire cooldown, and whether it fires *at all* can all be changed live via
`CombatEnv.set_enemy_difficulty()`, without rebuilding the environment. This exists because a
full-difficulty enemy turns out to be **unbeatable from a cold start** — an untrained policy
essentially never wins against instant, perfect return fire, so it never gets to practice the
aiming/engagement skills it would need to eventually win. `training/train.py`'s difficulty
**curriculum** uses this hook to start the enemy harmless (or firing only sluggishly) and ramp
it up in stages as the soldier's win rate against the *current* stage converges — see "Training
curriculum" below for the full mechanism, and `training_default.yaml`'s `train.curriculum`
comments for the empirical reasoning (including configurations that were tried and didn't
work) behind exactly how it's staged.

### Ballistics: how a fire attempt is actually resolved

This one mechanism resolves *every* shot in the simulation — the soldier's and the enemy's
alike, symmetrically, with no special-casing either direction. Given a fire attempt, in order:

1. **Ammo check.** No ammo → the attempt does nothing (no hit, no damage) and doesn't start a
   cooldown, since nothing was actually fired.
2. **Line-of-sight gate.** The shooter's current orientation bin must exactly equal the
   *target's true bearing bin* (i.e., the bin the target is actually standing in right now, not
   an approximation). If it doesn't match, the shot **cannot hit, at all** — this is a hard
   gate, not a probability penalty. It still consumes ammo and starts the cooldown, though —
   firing while facing the wrong way is a real, wasted shot, not a free action.
3. **Range check.** If the target is beyond the shooter's `max_range`, same result: guaranteed
   miss, but ammo/cooldown are still spent.
4. **Accuracy roll.** Only if LOS and range both pass: hit probability is read off the
   shooter's accuracy-vs-distance table at the exact current distance (linearly interpolated
   between the table's configured points, clamped beyond its ends), and a hit/miss is rolled
   against that probability.
5. **Damage.** If it hit, damage is read off a **second, independent** distance-vs-damage
   table — not derived from the accuracy number at all. Both curves get worse with range, so a
   close-in, correctly-aimed shot is both more likely to land and hits harder than the same
   shot taken from farther away.

Both combatants' shots are resolved through this exact process, independently, within the same
`step()` call — neither one's shot outcome depends on or is aware of the other's that tick.

### How an episode ends

Every episode starts with both combatants at full HP/ammo, both positions randomized (subject
to the minimum-separation rule above), and the soldier's starting orientation randomized too.
From there, it ends the instant any one of these becomes true:

- **Enemy HP reaches 0** → outcome `"win"`.
- **Soldier HP reaches 0** → outcome `"loss"`.
- **The tick count hits `env.episode.max_timesteps`** with both sides still alive → outcome
  `"timeout"` — tracked as its own distinct outcome, not folded into a win or a loss, since a
  cautious stalemate is a meaningfully different failure mode than actually dying.
- **Special case — simultaneous double-kill**: if both HPs hit 0 on the very same tick (both
  shots landing in the same `step()` call), it's scored as a **win**, not a draw. There's no
  fourth "draw" outcome anywhere in this codebase; enemy-death is checked before soldier-death
  specifically so this case falls out as a win. (This was an explicit design decision made
  during implementation — the original design spec never addressed the simultaneous-death case.)

### Action space (what the soldier's policy actually outputs)

The DQN doesn't output "move" and "fire" and "aim" separately — Q-learning needs one flat list
of discrete actions to take an `argmax` over, so all three decisions are packed into a single
integer:

| Component | Choices | Meaning |
|---|---|---|
| move direction | 9 | stay, or one of 8 compass directions |
| fire | 2 | attempt to fire this tick, or don't |
| orientation bin | `n_bins` (10 by default) | which way to face this tick |

`action_space = Discrete(9 * 2 * n_bins)` — **180 total actions** at the default 10-bin
setup. (The original design spec assumed 36 bins / 648 actions; the bin count was deliberately
coarsened during implementation specifically to make the line-of-sight gate reachable by chance
during early random exploration — a random action has a 1-in-10 shot at the correct orientation
bin instead of 1-in-36, which matters a lot before the policy has learned anything about
aiming. `env.bin_size_degrees` is a config value, so this is retunable without touching code —
finer bins are a straightforward way to make aiming harder/more realistic later.)

### Observation space (what the soldier's policy actually sees)

Every tick, the policy's input is one flat vector of `12 + 2*n_bins` floats (**32-dim** at the
default bin count) — no grid, no image, just numbers:

| Field(s) | Count | What it is |
|---|---|---|
| `dx, dy` | 2 | the enemy's position **relative to the soldier** (never absolute map coordinates) |
| soldier ammo, max range, cooldown remaining, accuracy-at-current-distance | 4 | the soldier's own weapon state — accuracy is handed over pre-computed from the lookup table, not left for the network to infer from raw distance |
| enemy ammo, max range, cooldown remaining, accuracy-at-current-distance | 4 | the same four fields, mirrored for the enemy (the soldier is assumed to be able to read/estimate the enemy's weapon state) |
| soldier HP, enemy HP | 2 | both units' current health |
| soldier orientation | `n_bins` | one-hot vector over the orientation bins |
| enemy orientation | `n_bins` | one-hot vector over the orientation bins |

Orientation is one-hot encoded rather than a raw angle specifically so the network never has to
learn circular wraparound on its own (that bin 9 is adjacent to bin 0, for instance) — as a
one-hot vector, adjacency isn't something the network needs to discover, it's just there in
the input.

### Reward structure (what actually shapes the learned behavior)

Every `step()` call returns one number (`_get_reward` in `combat_env.py`), which is just a sum
of independent terms added together — nothing multiplicative, nothing conditional on more than
one thing at once. At a glance, with the actual default weights from `training_default.yaml`'s
`reward:` section filled in:

```
reward = -0.03                                   # step_penalty, every tick, flat
        + 5.0   * damage_dealt_this_tick          # hit_reward_scale
        - 1.5   * damage_taken_this_tick          # damage_taken_penalty_scale
        + (-30.0 if ammo_just_hit_zero else 0)    # ammo_depleted_penalty, one-time
        + (0.01 if facing_enemy else 0)           # facing_enemy_reward, every tick
        + (0.5 if fired_while_facing else 0)      # fire_while_facing_bonus
        + (-0.5 if fired_while_not_facing else 0) # fire_while_not_facing_penalty
        + 0.05  * distance_closed_this_tick       # distance_closing_reward_scale, signed
        + {win: +300.0, lose: -50.0, timeout: -70.0}[outcome]   # only on the terminal tick
```

Everything below the step penalty is zero most ticks — a typical mid-episode tick where the
soldier moved a bit, was still cooling down, and wasn't facing the enemy yet is just
`-0.03 + 0.05 * distance_closed`. The terminal outcome term only ever applies once, on the
very last tick of an episode. The subsections below explain *why* each term looks the way it
does — most of them exist because an earlier, simpler version of the reward created an
unintended shortcut the policy learned to exploit instead of the intended behavior, and the
current shape is what closed that loophole:

- A small **flat penalty every tick**, just for time passing — discourages stalling.
- **Damage dealt** (scaled up) and **damage taken** (scaled down, i.e. penalized) — the direct
  "did something good/bad just happen" signal.
- A **one-time penalty the instant ammo hits zero** — a single meaningful penalty for emptying
  the mag, rather than a small penalty repeated on every subsequent empty-gun attempt (which
  didn't actually discourage the behavior; see the comment in `training_default.yaml` for the
  before/after reasoning).
- A **small dense reward for facing the enemy's true bearing**, every tick, whether or not it
  fires — deliberately kept smaller than the flat step penalty, so passively facing the enemy
  without ever shooting is still net-negative overall (just less negative than facing away).
  An earlier version of this bonus wasn't capped this way, and the policy learned to just stare
  at the enemy and farm the reward without ever engaging — this cap exists specifically to
  close that loophole.
- A **bonus for actually firing while correctly facing the enemy**, and a **matching penalty for
  firing while not facing it** (a guaranteed-miss shot) — these only trigger on genuine fire
  attempts with ammo, never for passive facing, which is what keeps the facing-reward above
  from being farmable on its own.
- A **signed distance-closing term**: positive when a tick's movement reduced the distance to
  the enemy, negative when it increased it. Over a full episode this telescopes down to just
  the *net* distance closed (start position vs. end position) — oscillating back and forth
  nets out to roughly zero, so it rewards genuine progress toward engagement range, not
  meaningless wiggling.
- **Large terminal bonuses/penalties** on episode end: a big bonus for winning, a big penalty
  for losing, and a smaller (but still negative) penalty for timing out — timing out is
  deliberately worse than winning but better than dying, since a stalemate at least didn't get
  the soldier killed.

### What's explicitly out of scope (for now)

Carried over unchanged from the original design spec: no reload mechanic (ammo is one fixed
pool per episode for both sides), no partial observability beyond the raw numbers already in
the observation, no enemy movement or more sophisticated enemy behavior (self-play, evasion,
etc.), no continuous/smooth orientation (bins are a deliberate choice to keep this a "pure"
discrete-action DQN problem), and no obstacles or map complexity.

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
