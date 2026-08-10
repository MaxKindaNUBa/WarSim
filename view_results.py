"""Plot mean eval reward (with confidence band) across seeds of a multi-seed run.

Usage:
    python view_results.py [run_name] [--metric avg_reward] [--ci 0.95] [--out path.png]

<run_name> is a folder under runs/ produced by training/train_multi_seed.py, i.e.
runs/<run_name>/<seed>/eval_log.csv for each seed (a plain runs/<run_name>/eval_log.csv
single-seed run also works, treated as one seed). If <run_name> is omitted, runs/ is
scanned and a picker window lets you choose one. This script reads every seed's
eval_log.csv, aligns them on checkpoint_step, and plots the across-seed mean with a
confidence interval band -- never the individual seed curves.
"""
import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def find_seed_logs(run_name):
    run_dir = Path(run_name)
    if not run_dir.is_dir():
        run_dir = RUNS_DIR / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_name} (looked in '.' and '{RUNS_DIR}')")

    logs = sorted(run_dir.glob("*/eval_log.csv"))
    if not logs and (run_dir / "eval_log.csv").is_file():
        logs = [run_dir / "eval_log.csv"]
    if not logs:
        raise FileNotFoundError(f"No eval_log.csv found under {run_dir}")
    return run_dir, logs


def scan_runs():
    """Return sorted [(run_dir, num_seeds), ...] for every folder under runs/ that
    contains an eval_log.csv, either directly or one level down in seed subfolders."""
    if not RUNS_DIR.is_dir():
        return []

    found = []
    for candidate in sorted(RUNS_DIR.iterdir()):
        if not candidate.is_dir():
            continue
        seed_logs = list(candidate.glob("*/eval_log.csv"))
        if seed_logs:
            found.append((candidate, len(seed_logs)))
        elif (candidate / "eval_log.csv").is_file():
            found.append((candidate, 1))

    # Newest first -- run folder names carry a trailing _YYYYMMDD_HHMMSS timestamp.
    found.sort(key=lambda item: item[0].name, reverse=True)
    return found


def select_run_gui(runs):
    """Show a picker window listing every discovered run; returns the chosen Path
    or None if the window was closed without a selection."""
    selected = {"run_dir": None}

    root = tk.Tk()
    root.title("Select a run to view")
    root.geometry("620x420")

    tk.Label(root, text="Select a run from runs/:", anchor="w").pack(fill="x", padx=10, pady=(10, 0))

    list_frame = tk.Frame(root)
    list_frame.pack(fill="both", expand=True, padx=10, pady=10)

    scrollbar = tk.Scrollbar(list_frame, orient="vertical")
    listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("TkFixedFont", 10))
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    for run_dir, num_seeds in runs:
        seed_label = f"{num_seeds} seed" + ("s" if num_seeds != 1 else "")
        listbox.insert("end", f"{run_dir.name}  ({seed_label})")
    if runs:
        listbox.selection_set(0)

    def confirm(event=None):
        selection = listbox.curselection()
        if selection:
            selected["run_dir"] = runs[selection[0]][0]
        root.destroy()

    listbox.bind("<Double-Button-1>", confirm)

    button_row = tk.Frame(root)
    button_row.pack(fill="x", padx=10, pady=(0, 10))
    ttk.Button(button_row, text="Cancel", command=root.destroy).pack(side="right")
    ttk.Button(button_row, text="View", command=confirm).pack(side="right", padx=(0, 8))

    root.mainloop()
    return selected["run_dir"]


def load_metric_matrix(logs, metric):
    per_seed = []
    for log_path in logs:
        df = pd.read_csv(log_path)
        if metric not in df.columns:
            raise ValueError(f"Column '{metric}' not found in {log_path} (columns: {list(df.columns)})")
        per_seed.append(df.set_index("checkpoint_step")[metric])

    # Outer-join on checkpoint_step so all seeds line up on the same x-axis
    # even if a run has extra/missing eval checkpoints.
    matrix = pd.concat(per_seed, axis=1)
    matrix.columns = [log_path.parent.name for log_path in logs]
    return matrix.sort_index()


def mean_and_ci(matrix, ci):
    n = matrix.count(axis=1)
    mean = matrix.mean(axis=1)
    sem = matrix.sem(axis=1)
    t_crit = stats.t.ppf((1 + ci) / 2, df=(n - 1).clip(lower=1))
    margin = t_crit * sem
    return mean, margin


RATE_METRICS = [
    ("win_rate", "tab:green"),
    ("loss_rate", "tab:red"),
    ("timeout_rate", "tab:orange"),
]


def plot_mean_ci(ax, logs, metric, ci, color, label=None):
    matrix = load_metric_matrix(logs, metric)
    mean = matrix.mean(axis=1)
    ax.plot(mean.index, mean.values, color=color, label=label or metric)
    if matrix.shape[1] > 1:
        _, margin = mean_and_ci(matrix, ci)
        ax.fill_between(
            mean.index, (mean - margin).values, (mean + margin).values,
            color=color, alpha=0.2,
        )
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", nargs="?", default=None,
                         help="Run folder name under runs/, e.g. dqn_v3_PER_curriculum_20260807_231210. "
                              "If omitted, a picker window opens to choose from runs/.")
    parser.add_argument("--metric", default="avg_reward", help="Column from eval_log.csv to plot (default: avg_reward)")
    parser.add_argument("--ci", type=float, default=0.9, help="Confidence level for the band (default: 0.95)")
    parser.add_argument("--out", default=None, help="Path to save the figure instead of showing it interactively")
    args = parser.parse_args()

    if args.run_name:
        run_dir, logs = find_seed_logs(args.run_name)
    else:
        runs = scan_runs()
        if not runs:
            raise FileNotFoundError(f"No runs with eval_log.csv found under {RUNS_DIR}")
        chosen = select_run_gui(runs)
        if chosen is None:
            print("No run selected, exiting.")
            return
        run_dir, logs = find_seed_logs(chosen)

    print(f"Found {len(logs)} seed run(s) under {run_dir}")

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

    top_label = (f"mean {args.metric} ({len(logs)} seeds)" if len(logs) > 1
                 else f"{args.metric} (1 seed, no CI)")
    plot_mean_ci(ax_top, logs, args.metric, args.ci, "tab:blue", label=top_label)
    ax_top.set_ylabel(args.metric)
    ax_top.set_title(f"{run_dir.name}: {args.metric} vs training step")
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    for metric, color in RATE_METRICS:
        plot_mean_ci(ax_bottom, logs, metric, args.ci, color)
    ax_bottom.set_xlabel("checkpoint_step")
    ax_bottom.set_ylabel("rate")
    ax_bottom.set_title("win / loss / timeout rate vs training step")
    ax_bottom.legend()
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
