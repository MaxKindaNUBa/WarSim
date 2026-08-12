"""Load a checkpoint (or a baseline policy) and watch it play in the pygame
GUI, paced to real time so playback matches timestep_duration_s.

Mechanism: record t_start at the top of each loop iteration; after
env.step() + env.render() complete, sleep for whatever's left of the tick
budget. Only ever slows down, never speeds up, so a slow render frame just
eats into the sleep budget instead of desyncing the sim.
"""

# python -m training.watch_policy --policy dqn --checkpoint runs/dqn_v1_20260806_233013/checkpoints/checkpoint_best.pt
# python -m training.watch_policy                      -> opens a picker GUI instead
import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import yaml

from agents.dqn import DQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from envs.combat_env import CombatEnv
from rendering import pygame_renderer

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _sim_defaults():
    return _load_yaml(CONFIG_DIR / "sim_default.yaml")


def _training_defaults():
    return _load_yaml(CONFIG_DIR / "training_default.yaml")


def _checkpoint_sort_key(path):
    name = path.stem
    if name == "checkpoint_best":
        return (0, 0)
    if name.startswith("checkpoint_step"):
        try:
            return (1, int(name[len("checkpoint_step"):]))
        except ValueError:
            pass
    return (2, name)


def list_checkpoints(seed_dir):
    ckpt_dir = Path(seed_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted(ckpt_dir.glob("*.pt"), key=_checkpoint_sort_key)


def seed_dirs_in_run(run_dir):
    """[seed_dir, ...] if run_dir holds multiple seed subfolders (each its own
    checkpoints/), else [] if run_dir itself is a single-seed run folder."""
    return sorted(p for p in Path(run_dir).iterdir() if p.is_dir() and (p / "checkpoints").is_dir())


def scan_runs():
    """Sorted (newest first) [run_dir, ...] under runs/ usable as a checkpoint source,
    either directly (single-seed run) or via seed subfolders (multi-seed run)."""
    if not RUNS_DIR.is_dir():
        return []
    found = [
        candidate for candidate in RUNS_DIR.iterdir()
        if candidate.is_dir() and ((candidate / "checkpoints").is_dir() or seed_dirs_in_run(candidate))
    ]
    found.sort(key=lambda p: p.name, reverse=True)
    return found


def _merged_over_defaults(defaults, overrides):
    # Top-level merge, not a deep one: older runs' config_snapshot.yaml can predate a config
    # key being renamed/added (e.g. ammo_depleted_penalty used to be empty_gun_fire_penalty),
    # so a raw snapshot dict can KeyError inside CombatEnv/DQNAgent.from_config. Falling back
    # to today's defaults for anything the snapshot doesn't have keeps old runs watchable;
    # reward_config in particular only affects the printed reward, never the policy's actions.
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


def _load_run_configs(config_snapshot_path):
    snapshot = _load_yaml(config_snapshot_path)
    env_config = _merged_over_defaults(_sim_defaults()["env"], snapshot.get("env"))
    reward_config = _merged_over_defaults(_training_defaults()["reward"], snapshot.get("reward"))
    train_config = _merged_over_defaults(_training_defaults()["train"], snapshot.get("train"))
    return env_config, reward_config, train_config


class _PickerGUI:
    def __init__(self, root):
        self.root = root
        self.result = None
        self.runs = scan_runs()

        root.title("Watch Policy")
        root.geometry("640x580")

        policy_frame = tk.LabelFrame(root, text="Policy")
        policy_frame.pack(fill="x", padx=10, pady=(10, 5))
        self.policy_var = tk.StringVar(value="dqn")
        for value, label in [("dqn", "DQN checkpoint"), ("heuristic", "Heuristic baseline"), ("random", "Random baseline")]:
            tk.Radiobutton(
                policy_frame, text=label, variable=self.policy_var, value=value, command=self._on_policy_change,
            ).pack(side="left", padx=8, pady=4)

        run_frame = tk.LabelFrame(root, text="Run (runs/)")
        run_frame.pack(fill="both", expand=True, padx=10, pady=5)
        list_container = tk.Frame(run_frame)
        list_container.pack(fill="both", expand=True, padx=8, pady=8)
        scrollbar = tk.Scrollbar(list_container, orient="vertical")
        self.run_listbox = tk.Listbox(
            list_container, yscrollcommand=scrollbar.set, font=("TkFixedFont", 10), exportselection=False,
        )
        scrollbar.config(command=self.run_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.run_listbox.pack(side="left", fill="both", expand=True)
        self.run_listbox.bind("<<ListboxSelect>>", self._on_run_select)
        for run_dir in self.runs:
            seeds = seed_dirs_in_run(run_dir)
            tag = f"{len(seeds)} seeds" if seeds else "1 seed"
            self.run_listbox.insert("end", f"{run_dir.name}  ({tag})")

        selector_frame = tk.Frame(root)
        selector_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(selector_frame, text="Seed:").grid(row=0, column=0, sticky="w")
        self.seed_var = tk.StringVar()
        self.seed_combo = ttk.Combobox(selector_frame, textvariable=self.seed_var, state="readonly", width=20)
        self.seed_combo.grid(row=0, column=1, sticky="w", padx=(4, 20))
        self.seed_combo.bind("<<ComboboxSelected>>", self._on_seed_select)
        tk.Label(selector_frame, text="Checkpoint:").grid(row=0, column=2, sticky="w")
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(selector_frame, textvariable=self.checkpoint_var, state="readonly", width=24)
        self.checkpoint_combo.grid(row=0, column=3, sticky="w", padx=4)

        options_frame = tk.LabelFrame(root, text="Playback options")
        options_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(options_frame, text="Episodes:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.episodes_var = tk.StringVar(value="5")
        tk.Entry(options_frame, textvariable=self.episodes_var, width=6).grid(row=0, column=1, sticky="w")
        tk.Label(options_frame, text="Env seed:").grid(row=0, column=2, sticky="w", padx=8)
        self.env_seed_var = tk.StringVar(value="0")
        tk.Entry(options_frame, textvariable=self.env_seed_var, width=6).grid(row=0, column=3, sticky="w")
        self.enemy_fire_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Enemy fires back", variable=self.enemy_fire_var).grid(row=0, column=4, padx=12)

        self.record_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            options_frame, text="Record to file:", variable=self.record_var, command=self._on_record_toggle,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 6))
        self.record_path_var = tk.StringVar(value="")
        self.record_entry = tk.Entry(options_frame, textvariable=self.record_path_var, width=32, state="disabled")
        self.record_entry.grid(row=1, column=1, columnspan=3, sticky="we", padx=4)
        self.record_browse_btn = ttk.Button(options_frame, text="Browse...", command=self._browse_record_path, state="disabled")
        self.record_browse_btn.grid(row=1, column=4, padx=8)

        self.status_label = tk.Label(root, text="", fg="firebrick", anchor="w")
        self.status_label.pack(fill="x", padx=10)
        button_row = tk.Frame(root)
        button_row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_row, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(button_row, text="Watch", command=self._confirm).pack(side="right", padx=(0, 8))

        self._on_policy_change()
        if self.runs:
            self.run_listbox.selection_set(0)
            self._on_run_select()

    def _on_policy_change(self):
        is_dqn = self.policy_var.get() == "dqn"
        self.run_listbox.config(state=tk.NORMAL if is_dqn else tk.DISABLED)
        self.seed_combo.config(state=("readonly" if is_dqn and len(self.seed_combo["values"]) > 1 else "disabled"))
        self.checkpoint_combo.config(state="readonly" if is_dqn else "disabled")

    def _selected_run(self):
        selection = self.run_listbox.curselection()
        if not selection or not self.runs:
            return None
        return self.runs[selection[0]]

    def _on_run_select(self, event=None):
        run_dir = self._selected_run()
        if run_dir is None:
            return
        seeds = seed_dirs_in_run(run_dir)
        if seeds:
            self.seed_combo["values"] = [p.name for p in seeds]
            self.seed_combo.current(0)
            self.seed_combo.config(state="readonly")
        else:
            self.seed_combo["values"] = [run_dir.name]
            self.seed_combo.current(0)
            self.seed_combo.config(state="disabled")  # single-seed run: nothing to pick
        self._on_seed_select()

    def _current_seed_dir(self):
        run_dir = self._selected_run()
        if run_dir is None:
            return None
        seeds = seed_dirs_in_run(run_dir)
        if not seeds:
            return run_dir
        index = self.seed_combo.current()
        return seeds[index if index >= 0 else 0]

    def _on_seed_select(self, event=None):
        seed_dir = self._current_seed_dir()
        checkpoints = list_checkpoints(seed_dir) if seed_dir else []
        self.checkpoint_combo["values"] = [p.name for p in checkpoints]
        if checkpoints:
            self.checkpoint_combo.current(0)  # checkpoint_best.pt sorts first

    def _on_record_toggle(self):
        state = "normal" if self.record_var.get() else "disabled"
        self.record_entry.config(state=state)
        self.record_browse_btn.config(state=state)

    def _browse_record_path(self):
        path = filedialog.asksaveasfilename(
            title="Save recording as",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("GIF animation", "*.gif"), ("All files", "*.*")],
        )
        if path:
            self.record_path_var.set(path)

    def _confirm(self):
        config_snapshot = None
        checkpoint = None
        if self.policy_var.get() == "dqn":
            seed_dir = self._current_seed_dir()
            if seed_dir is None or not self.checkpoint_var.get():
                self.status_label.config(text="Select a run and checkpoint first.")
                return
            checkpoint = seed_dir / "checkpoints" / self.checkpoint_var.get()
            config_snapshot = seed_dir / "config_snapshot.yaml"
            if not config_snapshot.is_file():
                config_snapshot = None

        try:
            episodes = int(self.episodes_var.get())
            env_seed = int(self.env_seed_var.get())
        except ValueError:
            self.status_label.config(text="Episodes and env seed must be integers.")
            return

        record_path = None
        if self.record_var.get():
            record_path = self.record_path_var.get().strip()
            if not record_path:
                self.status_label.config(text="Choose a path to record to, or uncheck Record to file.")
                return

        self.result = {
            "policy": self.policy_var.get(),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "config_snapshot": config_snapshot,
            "episodes": episodes,
            "seed": env_seed,
            "enemy_fire_enabled": self.enemy_fire_var.get(),
            "record_path": record_path,
        }
        self.root.destroy()

    def _cancel(self):
        self.result = None
        self.root.destroy()


def select_watch_options_gui():
    """Picker window for choosing a policy / run / seed / checkpoint under runs/, plus
    playback options. Returns a dict of watch() kwargs, or None if the window was closed
    without confirming."""
    root = tk.Tk()
    app = _PickerGUI(root)
    root.mainloop()
    return app.result


def _episode_record_path(record_path, ep, episodes):
    """record_path as-is for a single-episode watch; otherwise suffixed with _epN so
    each episode (each of which starts with an env.reset(), a visible discontinuity
    a video can't represent mid-clip) gets its own file."""
    if episodes == 1:
        return record_path
    return record_path.with_name(f"{record_path.stem}_ep{ep}{record_path.suffix}")


def watch(policy, checkpoint=None, episodes=5, seed=0, enemy_fire_enabled=True,
          env_config=None, reward_config=None, render_config=None,
          train_config=None, agent_config=None, record_path=None, record_fps=None):
    env_config = env_config or _sim_defaults()["env"]
    reward_config = reward_config or _training_defaults()["reward"]
    render_config = render_config or _sim_defaults()["render"]
    agent_config = agent_config or _sim_defaults()["agent"]

    env = CombatEnv(
        env_config=env_config, reward_config=reward_config, render_config=render_config,
        render_mode="human", enemy_fire_enabled=enemy_fire_enabled,
    )

    if policy == "random":
        agent = RandomAgent(env.action_space)
        act = lambda obs: agent.act(obs)
    elif policy == "heuristic":
        agent = HeuristicAgent(
            env.n_bins, env.bin_size_degrees,
            preferred_range=agent_config["heuristic"]["preferred_range"],
        )
        act = lambda obs: agent.act(obs)
    elif policy == "dqn":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --policy dqn")
        train_config = train_config or _training_defaults()["train"]
        agent = DQNAgent.from_config(env.observation_space.shape[0], env.action_space.n, train_config)
        agent.load_checkpoint(checkpoint)
        act = lambda obs: agent.act(obs, global_step=0, greedy=True)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    timestep_duration_s = env.timestep_duration_s

    record_path = Path(record_path) if record_path else None
    if record_path is not None:
        import imageio  # deferred: only needed when recording
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fps = record_fps or round(1.0 / timestep_duration_s)

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            terminated = truncated = False

            writer = None
            ep_path = None
            if record_path is not None:
                ep_path = _episode_record_path(record_path, ep, episodes)
                writer = imageio.get_writer(str(ep_path), fps=fps, macro_block_size=1)

            while not (terminated or truncated):
                t_start = time.perf_counter()
                action = act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if writer is not None:
                    writer.append_data(pygame_renderer.get_frame())
                elapsed = time.perf_counter() - t_start
                time.sleep(max(0.0, timestep_duration_s - elapsed))

            if writer is not None:
                writer.close()
                print(f"Episode {ep}: outcome={info['outcome']} (saved {ep_path})")
            else:
                print(f"Episode {ep}: outcome={info['outcome']}")
    finally:
        env.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        options = select_watch_options_gui()
        if options is None:
            print("No selection made, exiting.")
            sys.exit(0)
        env_config = reward_config = train_config = None
        if options["config_snapshot"] is not None:
            env_config, reward_config, train_config = _load_run_configs(options["config_snapshot"])
        watch(
            policy=options["policy"], checkpoint=options["checkpoint"], episodes=options["episodes"],
            seed=options["seed"], enemy_fire_enabled=options["enemy_fire_enabled"],
            env_config=env_config, reward_config=reward_config, train_config=train_config,
            record_path=options["record_path"],
        )
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "heuristic", "dqn"], default="heuristic")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-enemy-fire", action="store_true")
    parser.add_argument("--env-config", default=None,
                         help="Path to an env config YAML (default: env: section of config/sim_default.yaml)")
    parser.add_argument("--reward-config", default=None,
                         help="Path to a reward config YAML (default: reward: section of config/training_default.yaml)")
    parser.add_argument("--render-config", default=None,
                         help="Path to a render config YAML (default: render: section of config/sim_default.yaml)")
    parser.add_argument("--train-config", default=None,
                         help="Path to a train config YAML — must match the checkpoint's architecture, "
                              "only used with --policy dqn (default: train: section of config/training_default.yaml)")
    parser.add_argument("--agent-config", default=None,
                         help="Path to an agent config YAML for the heuristic policy "
                              "(default: agent: section of config/sim_default.yaml)")
    parser.add_argument("--record", default=None,
                         help="Save the rendered episode(s) as video/GIF to this path "
                              "(format inferred from extension, e.g. .mp4 or .gif). "
                              "With --episodes > 1, each episode is saved to its own "
                              "_epN-suffixed file.")
    parser.add_argument("--record-fps", type=int, default=10,
                         help="Frame rate for --record output (default: 1 / timestep_duration_s, "
                              "i.e. real-time playback speed)")
    args = parser.parse_args()
    watch(
        policy=args.policy, checkpoint=args.checkpoint, episodes=args.episodes,
        seed=args.seed, enemy_fire_enabled=not args.no_enemy_fire,
        env_config=_load_yaml(args.env_config) if args.env_config else None,
        reward_config=_load_yaml(args.reward_config) if args.reward_config else None,
        render_config=_load_yaml(args.render_config) if args.render_config else None,
        train_config=_load_yaml(args.train_config) if args.train_config else None,
        agent_config=_load_yaml(args.agent_config) if args.agent_config else None,
        record_path=args.record, record_fps=args.record_fps,
    )
