# DQN Combat Simulation — Implementation Action Plan

**Status:** Environment design finalized. This document plans the actual codebase build, starting from an empty repo.

---

## 1. Confirmed Design Spec (reference)

- **Action space (flattened discrete):** `move_direction (9)` × `fire (2)` × `orientation (N_BINS)`. `N_BINS = 360 / bin_size_degrees`, default `bin_size_degrees = 10` → 36 bins → **648 total actions**.
- **Observation space:** `dx, dy`; soldier gun (ammo, max_range, cooldown_remaining, **current accuracy at present distance**); enemy gun (ammo, max_range, cooldown_remaining, **current accuracy at present distance**); soldier HP; enemy HP; soldier orientation (one-hot, `N_BINS`); enemy orientation (one-hot, `N_BINS`). The accuracy fields are computed each step via `ballistics.accuracy_lookup(distance, table)` and injected directly into the obs vector — the agent shouldn't have to infer its own hit odds purely from `dx,dy`, since it's meant to reason about "is this a good range to be firing from" directly.
- **Fire action is always legal**, even at 0 ammo — attempting to fire with an empty gun is allowed (future bullet-awareness hook) and penalized via reward, not blocked by the environment.
- **No reload.** Fixed `max_ammo` per episode for both agents.
- **Two independent lookup tables:** accuracy(distance), damage(distance). Fixed placeholder values now, tunable later, loaded from config — not hardcoded.
- **LOS-gated hits:** shot only has nonzero hit probability if shooter's orientation bin matches the true bearing bin to target. No-LOS shots still consume ammo and trigger cooldown.
- **Enemy:** cannot move; instantly orients to face soldier every step (binned); fires if in range and cooldown allows. No turn lag.
- **Map:** bounded (fixed size, for sim speed only), no obstacles. Soldier moves in 8 directions + stay, fixed step size.
- **Discrete timesteps:** the simulation advances in fixed discrete ticks. The real-world duration represented by one tick (`timestep_duration_s`) is a config value, not hardcoded — it scales how cooldowns, movement step size, and (later) any rate-based mechanics are interpreted, without changing the step-based control loop itself. **Pacing is tied to render mode, not a separate flag:** in GUI mode (`render_mode="human"`), each `step()` call is throttled to real wall-clock time so one tick takes `timestep_duration_s` seconds to play out on screen (see Section 5 for the mechanism). In headless mode (`render_mode=None`, i.e. all training and bulk evaluation), there is **no throttling at all** — `step()` is called back-to-back as fast as the CPU/GPU can execute, so the simulated clock ticks far faster than real time, maximizing training throughput on your laptop.
- **Episode:** random soldier + enemy spawn each reset (min separation constraint), random soldier start orientation. Terminates on soldier death, enemy death, or max timestep (timeout — logged as distinct outcome).

---

## 2. Tech Stack / Libraries

**Python version: 3.12** (pin in a `pyproject.toml` / `.python-version` — verify `torch`'s current stable release supports 3.12 at setup time, as PyTorch's newest-Python support sometimes lags a few months behind a new CPython release).

| Purpose | Library | Notes |
|---|---|---|
| Env API | `gymnasium` | Standard `Env` interface — `reset()`, `step()`, `render()`. Keeps you compatible with existing tooling even though DQN is hand-written. |
| Numerics | `numpy` | State vectors, lookup table interpolation (`np.interp`), vector math for bearing/distance. Runs on CPU — this is fine, none of it is a training bottleneck. |
| DQN / training | `torch` (CUDA build) | Hand-written DQN — full control, matches "from scratch" goal. Install the CUDA-enabled wheel (`pip install torch --index-url https://download.pytorch.org/whl/cu121` or current CUDA channel) so it targets the RTX 4060. **All network weights, replay buffer tensors, and training math run on `cuda`** — see Section 2a. |
| Config management | `pyyaml` (or `omegaconf`) | All tunable params (bin size, map bounds, reward weights, lookup tables, hyperparameters, timestep duration) live in YAML, not hardcoded. |
| Structured logging | `logging` (stdlib) | Human-readable run logs, errors, warnings — separate from metrics. CPU-only, never touches the GPU. |
| Metrics/experiment tracking | `tensorboard` (via `torch.utils.tensorboard`) | Scalar curves: reward, loss, epsilon, win rate, hit rate. Free, local, no account needed. Writer itself is CPU/disk I/O — see Section 2a for how to keep it from touching CUDA even though it lives inside the `torch` package. |
| Optional richer tracking | `wandb` | Only if you want hosted dashboards/run comparison later — treat as optional, not required for v1. |
| GUI rendering | `pygame` | Lightweight, simple 2D primitives (circles, lines for orientation, HP bars) — ideal for this kind of top-down 2D sim. Runs in both windowed (GUI mode) and can be skipped entirely (headless mode). Does **not** use CUDA/PyTorch GPU compute — see Section 2a for how it's actually assigned a GPU. |
| Data logging (episodes/runs) | `csv` (stdlib) + `json` (stdlib) | Per-episode and per-run structured logs — see Section 6. CPU-only. |
| Plotting (post-hoc analysis) | `matplotlib` | Trajectory visualization, reward curves, evaluation summary plots. |
| Testing | `pytest` | Unit tests for env mechanics (bearing math, LOS gating, lookup interpolation) — critical given how easy this is to get subtly wrong. |

Install baseline:
```bash
pip install gymnasium numpy pyyaml tensorboard pygame matplotlib pytest
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or whatever CUDA channel matches your installed CUDA/driver version
```

---

## 2a. GPU / Device Strategy (RTX 4060 + Intel UHD)

Your machine has two GPUs: the RTX 4060 (CUDA-capable, discrete) and the Intel UHD (integrated). Here's how each part of the codebase is assigned, and what's actually controllable from Python versus what's an OS/driver setting.

**Training — fully on CUDA (RTX 4060), no ambiguity here:**
- `DQNAgent` holds `self.device = torch.device("cuda")`, set once at startup (fail loudly with `torch.cuda.is_available()` check if CUDA isn't detected, rather than silently falling back to CPU).
- `QNetwork` and target network both `.to(device)` immediately after construction.
- Replay buffer sampling returns tensors already moved to `device` before they hit the forward/backward pass — this is a common accidental CPU bottleneck (sampling on CPU is fine; leaving the sampled batch on CPU into the forward pass is not).
- Every tensor entering `train_step()` (states, actions, rewards, next_states, dones) is `.to(device)`.

**Environment / GUI rendering — does not use CUDA, by construction:**
- `pygame` renders via SDL2, which talks to whichever GPU the **operating system** assigns to that process — this is not something `pygame`'s Python API lets you select directly, and it's a separate rendering pipeline from PyTorch/CUDA entirely (pygame never imports or touches `torch.cuda`).
**Enforced in code, not left to OS configuration:** `pygame_renderer.py` sets `os.environ["SDL_RENDER_DRIVER"] = "software"` **before** `pygame.init()` is ever called. This forces SDL2 to rasterize everything (circles, lines, HP bars) on the CPU, so the render path never engages either GPU's 3D/compute pipeline at all — not the RTX 4060, not the Intel UHD. This is a hard guarantee baked into the module itself, not something that depends on OS settings, drivers, or the user remembering to configure anything, which is what "no risk" actually requires. Given the render workload here (a handful of primitives at low resolution, modest framerate), software rasterization has no perceptible performance cost.

- One caveat that's physically unavoidable and not a real risk: getting the finished frame onto the monitor still requires *some* GPU to scan it out to the display — that final hand-off is owned by the OS/display driver, not by pygame or PyTorch, and no application-level code can fully override it. To also steer that residual scanout toward the Intel card specifically (belt-and-suspenders, not required for the "no CUDA contention" guarantee above, which the software-renderer setting already covers):
  - **Windows:** Settings → System → Display → Graphics → add the Python interpreter (or a dedicated launcher `.exe`/shortcut for `watch_policy.py`) and set its preference to **"Power saving"** (Intel UHD).
  - **Linux:** launch the GUI script with `DRI_PRIME=0 python training/watch_policy.py` (Mesa/PRIME offload convention).
- Net effect: the render path is guaranteed by code to never touch CUDA or engage GPU compute on either card, regardless of whether the OS-level display-scanout steps above are ever set up.

**Logging — kept strictly off the CUDA GPU:**
- Anything scalar being logged (loss value, reward, epsilon) must be `.item()` or `.detach().cpu().numpy()`'d **before** it reaches `logging`, `csv`, `json`, or `SummaryWriter.add_scalar()` calls — never pass a live CUDA tensor into a logging call.
- `torch.utils.tensorboard.SummaryWriter` itself only performs disk I/O (writing event files) — it has no GPU compute path, so once inputs are detached to CPU scalars/numpy, the whole logging stack (`logging_utils/`, CSV writers, tensorboard writer) runs purely on CPU threads.
- If logging ever becomes a training-loop bottleneck (unlikely at this project's scale, but worth knowing), the fix is a background `threading.Thread` or `queue.Queue`-based async writer in `run_logger.py` — not moving any logging work onto CUDA.

---

## 3. Repository Structure

```
dqn-combat-sim/
├── config/
│   ├── env_default.yaml          # map bounds, bin size, ammo, gun stats, spawn rules
│   ├── reward_default.yaml       # all reward weights/penalties
│   └── train_default.yaml        # DQN hyperparameters
├── envs/
│   ├── __init__.py
│   ├── combat_env.py             # CombatEnv(gym.Env) — reset/step/observation
│   ├── geometry.py               # bearing, distance, angle-to-bin, bin-to-angle
│   ├── ballistics.py             # accuracy/damage lookup tables + interpolation, LOS check, hit resolution
│   └── spawner.py                # randomized spawn logic with min-separation constraint
├── agents/
│   ├── __init__.py
│   ├── dqn.py                    # QNetwork (MLP), DQNAgent (act/train/update_target)
│   ├── replay_buffer.py
│   ├── random_agent.py           # baseline
│   └── heuristic_agent.py        # baseline
├── training/
│   ├── train.py                  # main training loop entrypoint
│   └── evaluate.py               # fixed-seed evaluation entrypoint
├── rendering/
│   ├── __init__.py
│   └── pygame_renderer.py        # GUI mode renderer, consumed by combat_env when render_mode="human"
├── logging_utils/
│   ├── __init__.py
│   ├── run_logger.py             # sets up stdlib logging + tensorboard writer, per-run log dir
│   └── episode_logger.py         # per-episode CSV/JSON writer
├── tests/
│   ├── test_geometry.py
│   ├── test_ballistics.py
│   └── test_env_reset_step.py
├── notebooks/
│   └── analysis.ipynb            # post-hoc trajectory/metric visualization
├── runs/                         # gitignored — all training outputs land here (see Section 6)
├── requirements.txt
└── README.md
```

---

## 4. Build Order (phased, each phase independently testable)

### Phase 0 — Repo + config scaffolding
- Set up structure above, `requirements.txt`, empty stub files.
- Write `config/env_default.yaml` with all placeholder constants: map size, `bin_size_degrees`, `max_ammo`, `max_range`, `fire_cooldown_steps`, min spawn separation, max episode timesteps.
- Write `config/reward_default.yaml`: hit reward scale, damage-taken penalty scale, step penalty, terminal win/lose/timeout rewards, empty-gun-fire penalty, orientation-shaping bonus weight.

### Phase 1 — Geometry + ballistics (pure functions, no env yet)
- `geometry.py`: `bearing(p1, p2)`, `distance(p1, p2)`, `angle_to_bin(angle, bin_size)`, `bin_to_angle_center(bin_idx, bin_size)`.
- `ballistics.py`: `accuracy_lookup(distance, table)`, `damage_lookup(distance, table)` (both `np.interp` against config-defined breakpoints), `check_los(shooter_orientation_bin, true_bearing_bin)`, `resolve_shot(shooter_state, target_state, table_config) -> (hit: bool, damage: float, ammo_consumed: bool)`.
- **Write `tests/test_geometry.py` and `tests/test_ballistics.py` immediately** — bearing/binning off-by-ones are the single most likely source of silent bugs in this whole project. Test wraparound (0°/360° boundary), bin edges, and LOS true/false cases explicitly before moving on.

### Phase 2 — `CombatEnv` core (headless first, no rendering yet)
- Implement `reset()`: random spawn via `spawner.py`, reset HP/ammo/orientation, build initial observation.
- Implement `step(action)`: decode flattened action index → `(move_dir, fire, orientation_bin)`, apply movement (clip to bounds), resolve simultaneous soldier/enemy fire via `ballistics.resolve_shot`, apply damage, compute reward, check termination, build next observation.
- Implement `_get_obs()`, `_get_reward()`, `_check_terminated()` as clearly separated methods (not inlined in `step`) — makes unit testing and later reward-iteration much easier.
- `render_mode` param accepted (`None` / `"human"`) but Phase 2 only needs `None` (headless) to work — GUI comes in Phase 5.
- `tests/test_env_reset_step.py`: verify obs shape/dtype consistency, verify episode terminates correctly on HP depletion and on timeout, verify ammo decrements on every fire attempt (including empty-gun attempts).

### Phase 3 — Baseline agents + environment validation
- `random_agent.py`: samples uniformly from the 648 actions.
- `heuristic_agent.py`: scripted range-management + always-face-enemy-bearing + fire-when-ready logic (as previously discussed).
- Run both for a few thousand episodes headless, log reward/win-rate/hit-rate distributions. **Expect random agent hit rate to be very low** given LOS gating with 36 bins — treat a suspiciously high random-agent hit rate as a bug signal, not a good outcome.
- This phase is a gate: don't proceed to DQN until baseline numbers look sane.

### Phase 4 — DQN agent + training loop
- `replay_buffer.py`: simple circular buffer, `(s, a, r, s', done)` tuples.
- `dqn.py`: MLP `QNetwork` (widen given 648-action output — e.g., 3 hidden layers, 256 units), `DQNAgent` with epsilon-greedy `act()`, Double DQN target computation, `train_step()`, target network sync (hard update every N steps, or soft/Polyak — pick one, hard is simpler to start).
- `train.py`: main loop — step env, store transition, train every step (or every K steps), decay epsilon, periodic target sync, periodic checkpoint save, periodic evaluation rollout against fixed seeds.
- Start with **enemy firing disabled** (config flag) to confirm the soldier learns to close range and orient correctly before adding the harder simultaneous-combat dynamic.

### Phase 5 — GUI mode (pygame)
- `pygame_renderer.py`: given the env's internal state (positions, HP, orientation, bins), draw:
  - Soldier + enemy as circles (color-coded), HP bars above each
  - Orientation as a short line/cone from each unit in the direction of its current bin
  - Bounded map border
  - A faded circle showing effective max range
  - An on-hit flash/tracer line between shooter and target for the frame a hit lands
- `CombatEnv.render()` dispatches to `pygame_renderer` only when `render_mode="human"`; headless runs (training) never import/init pygame at all, so training performance is unaffected.
- Add a small standalone `training/watch_policy.py` script: loads a checkpoint, runs episodes with `render_mode="human"` at a throttled framerate, for visually inspecting a trained (or baseline) policy. This is where you'll actually *see* whether orientation/LOS logic looks correct — very useful debugging tool, build it early enough to help debug Phase 2/4, not just as a final demo.

### Phase 6 — Evaluation
- `evaluate.py`: fixed set of seeds (spawn pairs + start orientations), run N episodes per checkpoint, no exploration (`epsilon=0`), report total/average reward, win/loss/timeout rates, avg steps-to-kill, avg soldier HP remaining, hit-rate %, ammo-efficiency, empty-gun-fire attempt count.
- Compare trained DQN vs. random baseline vs. heuristic baseline on identical seed sets.

---

## 5. GUI vs. Headless Mode — design summary

- Single `CombatEnv(render_mode=None | "human")` constructor flag, standard Gymnasium convention.
- **Headless (`None`):** no pygame import/init at all — used for all training and bulk evaluation. Zero rendering overhead, and no time-based throttling: `step()` is called in a tight loop, so the discrete-tick simulation clock advances as fast as the machine allows, independent of `timestep_duration_s`.
- **GUI (`"human"`):** pygame window opens on first `render()` call, updates each step, **and the control loop paces itself to real time** so playback matches `timestep_duration_s`. Mechanism: record `t_start = time.perf_counter()` at the top of each loop iteration in `watch_policy.py`; after `env.step()` + `env.render()` complete, compute elapsed time and `time.sleep(max(0.0, timestep_duration_s - elapsed))` before starting the next tick. This is a simple fixed-timestep game-loop pattern — it only ever sleeps to slow down, never speeds up, so a slow render frame just eats into the sleep budget rather than desyncing the sim clock. Used for: manual debugging, watching a trained policy play at a human-readable pace, sanity-checking geometry/LOS visually.
- Keep `pygame_renderer.py` fully decoupled from `combat_env.py`'s core logic — the renderer only ever *reads* env state, never influences `step()` behavior, and the real-time pacing lives in the calling script (`watch_policy.py`), not inside `CombatEnv` itself. This means headless training is guaranteed bit-identical whether or not the renderer module is even importable in the environment (useful if you ever train on a headless server with no display), and `CombatEnv.step()` always executes at full speed regardless of render mode — pacing is purely a property of the loop that calls it.

---

## 6. Logging Strategy (training + results)

Every run gets its own timestamped directory: `runs/<run_name>_<timestamp>/`, containing:

```
runs/dqn_v1_20260806_143012/
├── config_snapshot.yaml     # exact merged config used for this run (for reproducibility)
├── run.log                  # stdlib logging output — human-readable, INFO+ level
├── tensorboard/             # scalar curves: reward, loss, epsilon, win/loss/timeout rate, hit rate
├── checkpoints/
│   ├── checkpoint_ep1000.pt
│   ├── checkpoint_ep2000.pt
│   └── checkpoint_best.pt   # best eval win-rate so far
├── episode_log.csv          # one row per training episode: episode #, total reward, length, outcome, hit rate, ammo used
├── eval_log.csv             # one row per evaluation pass: checkpoint step, total reward, win/loss/timeout %, avg steps-to-kill, avg HP remaining
└── final_summary.json       # end-of-run rollup stats
```

**Logging rules to build in from day one:**
- `run_logger.py` sets up both the stdlib `logging` handler (writes `run.log`, also echoes to console) and the `SummaryWriter` (tensorboard) in one call at run start — every other module just calls `logging.getLogger(__name__)`, never configures logging itself.
- `episode_logger.py` appends one CSV row **every episode**, not just periodically — cheap to write, and having the full per-episode history (not just periodic samples) makes later debugging of "when exactly did training destabilize" possible.
- Log **every exception** from the training loop with full traceback via `logging.exception(...)` before any crash — don't let a bare `try/except: pass` swallow errors silently, this environment has enough moving parts (geometry, binning, lookup tables) that silent failures are the main risk.
- `config_snapshot.yaml` gets written once at run start — the *fully resolved* config (defaults + any CLI overrides merged), so a run is always reproducible from its own output directory alone, without needing to know what config file/flags were originally used.
- Checkpoint on a fixed episode interval **and** whenever eval win-rate improves (`checkpoint_best.pt`) — gives you both a training-progression trail and a "best model so far" pointer.

---

## 7. Suggested Order of Work (condensed checklist)

1. Repo scaffold + config YAMLs
2. `geometry.py` + `ballistics.py` + their unit tests
3. `CombatEnv` (headless) + its unit tests
4. Random + heuristic baseline agents, validate env sanity
5. `run_logger.py` / `episode_logger.py` wired in from the start (not bolted on later)
6. DQN agent + replay buffer + `train.py` (enemy-fire disabled first)
7. Enable enemy fire, retrain, iterate on reward shaping
8. `pygame_renderer.py` + `watch_policy.py` — use this to visually debug 6–7 if anything looks off
9. `evaluate.py` + fixed-seed benchmark comparison across DQN/heuristic/random
10. Tune lookup tables (accuracy/damage curves) using accumulated eval data as your feedback loop

---

## Open items still deferred (explicitly out of scope for v1, noted for future work)
- Reload mechanic (fixed ammo only for now)
- True bullet-awareness / partial observability of own ammo state beyond the raw count
- Enemy movement or more complex enemy policies (self-play, scripted evasion)
- Smooth (non-binned) orientation or continuous action space
- Obstacles / map complexity
