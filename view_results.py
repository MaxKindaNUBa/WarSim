"""Plot eval metrics (mean with confidence band across seeds) for single or multiple runs.

Usage:
    # Interactive GUI picker (supports selecting 1 or multiple runs):
    python view_results.py

    # Single run plot (2 subplots: metric on top, win/loss/timeout rates on bottom):
    python view_results.py <run_name> [--metric avg_reward] [--ci 0.95] [--out path.png]

    # Multiple runs comparison (4 subplots: metric, win rate, loss rate, timeout rate):
    python view_results.py <run1> <run2> <run3> ... [--metric avg_reward] [--ci 0.95] [--out path.png]

<run_name> is a folder under runs/ produced by training/train_multi_seed.py, i.e.
runs/<run_name>/<seed>/eval_log.csv for each seed (a plain runs/<run_name>/eval_log.csv
single-seed run also works, treated as 1 seed).
"""
import argparse
import re
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
    """Show a picker window listing every discovered run with multi-selection support;
    returns a list of chosen Path objects or an empty list if cancelled."""
    selected_runs = []

    root = tk.Tk()
    root.title("Select Runs to View / Compare")
    root.geometry("700x520")

    header_frame = tk.Frame(root)
    header_frame.pack(fill="x", padx=10, pady=(10, 5))
    tk.Label(
        header_frame,
        text="Select one or multiple runs from runs/ (Ctrl+Click or Shift+Click):",
        font=("TkDefaultFont", 10, "bold"),
        anchor="w",
    ).pack(side="top", fill="x")
    tk.Label(
        header_frame,
        text="• 1 run selected: single experiment detailed view (2 graphs)\n"
             "• 2+ runs selected: comparison view across all selected experiments (4 graphs)",
        font=("TkDefaultFont", 9),
        fg="#555555",
        anchor="w",
        justify="left",
    ).pack(side="top", fill="x", pady=(2, 0))

    list_frame = tk.Frame(root)
    list_frame.pack(fill="both", expand=True, padx=10, pady=5)

    scrollbar = tk.Scrollbar(list_frame, orient="vertical")
    listbox = tk.Listbox(
        list_frame,
        selectmode=tk.EXTENDED,
        yscrollcommand=scrollbar.set,
        font=("TkFixedFont", 10),
    )
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    for run_dir, num_seeds in runs:
        seed_label = f"{num_seeds} seed" + ("s" if num_seeds != 1 else "")
        listbox.insert("end", f"{run_dir.name}  ({seed_label})")

    if runs:
        listbox.selection_set(0)

    # Quick selection shortcut bar
    quick_bar = tk.Frame(root)
    quick_bar.pack(fill="x", padx=10, pady=(0, 5))
    tk.Label(quick_bar, text="Quick select:", font=("TkDefaultFont", 9)).pack(side="left", padx=(0, 5))

    def select_top_n(n):
        listbox.selection_clear(0, "end")
        limit = min(n, len(runs))
        for i in range(limit):
            listbox.selection_set(i)

    def select_all():
        listbox.selection_set(0, "end")

    def clear_all():
        listbox.selection_clear(0, "end")

    ttk.Button(quick_bar, text="Past 5", command=lambda: select_top_n(5)).pack(side="left", padx=2)
    ttk.Button(quick_bar, text="Past 10", command=lambda: select_top_n(10)).pack(side="left", padx=2)
    ttk.Button(quick_bar, text="Select All", command=select_all).pack(side="left", padx=2)
    ttk.Button(quick_bar, text="Clear", command=clear_all).pack(side="left", padx=2)

    def confirm(event=None):
        selection = listbox.curselection()
        if not selection and runs:
            # Default to first item if nothing explicitly highlighted
            selection = (0,)
        for idx in selection:
            selected_runs.append(runs[idx][0])
        root.destroy()

    def double_click(event):
        # On double click, view only the clicked run
        selection = listbox.curselection()
        if selection:
            selected_runs.clear()
            selected_runs.append(runs[selection[0]][0])
            root.destroy()

    listbox.bind("<Double-Button-1>", double_click)

    button_row = tk.Frame(root)
    button_row.pack(fill="x", padx=10, pady=(5, 10))
    ttk.Button(button_row, text="Cancel", command=root.destroy).pack(side="right")
    ttk.Button(button_row, text="View Selected", command=confirm).pack(side="right", padx=(0, 8))

    root.mainloop()
    return selected_runs


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
    matrix.columns = [f"{log_path.parent.name}_{i}" for i, log_path in enumerate(logs)]
    return matrix.sort_index()


def mean_and_ci(matrix, ci):
    n = matrix.count(axis=1)
    mean = matrix.mean(axis=1)
    sem = matrix.sem(axis=1).fillna(0)
    t_crit = stats.t.ppf((1 + ci) / 2, df=(n - 1).clip(lower=1))
    margin = (t_crit * sem).fillna(0)
    return mean, margin


RATE_METRICS = [
    ("win_rate", "tab:green"),
    ("loss_rate", "tab:red"),
    ("timeout_rate", "tab:orange"),
]


def get_display_name(run_name):
    """Strip trailing _YYYYMMDD_HHMMSS timestamp from run name if present."""
    name = Path(run_name).name
    return re.sub(r'_\d{8}_\d{6}$', '', name)


def get_color(idx, total):
    """Return a visually distinct color for experiment index `idx` out of `total`."""
    if total <= 10:
        return plt.cm.tab10(idx % 10)
    elif total <= 20:
        return plt.cm.tab20(idx % 20)
    else:
        return plt.cm.turbo(idx / max(total - 1, 1))


def plot_mean_ci(ax, logs, metric, ci, color, label=None):
    matrix = load_metric_matrix(logs, metric)
    mean = matrix.mean(axis=1)
    ax.plot(mean.index, mean.values, color=color, label=label or metric, linewidth=1.8)
    if matrix.shape[1] > 1:
        _, margin = mean_and_ci(matrix, ci)
        ax.fill_between(
            mean.index, (mean - margin).values, (mean + margin).values,
            color=color, alpha=0.2,
        )
    return mean


def plot_single_run(run_dir, logs, metric="avg_reward", ci=0.95):
    """Plot single experiment detailed view (2 vertically stacked subplots)."""
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    display_name = get_display_name(run_dir.name)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"Run: {display_name}")

    top_label = (f"mean {metric} ({len(logs)} seeds)" if len(logs) > 1
                 else f"{metric} (1 seed, no CI)")
    plot_mean_ci(ax_top, logs, metric, ci, "tab:blue", label=top_label)
    ax_top.set_ylabel(metric)
    ax_top.set_title(f"{display_name}: {metric} vs training step")
    ax_top.legend(fontsize="small", loc="best")
    ax_top.grid(alpha=0.3)

    for rate_metric, color in RATE_METRICS:
        plot_mean_ci(ax_bottom, logs, rate_metric, ci, color)
    ax_bottom.set_xlabel("checkpoint_step")
    ax_bottom.set_ylabel("rate")
    ax_bottom.set_title("win / loss / timeout rate vs training step")
    ax_bottom.legend(fontsize="small", loc="best")
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_multi_runs(runs_data, metric="avg_reward", ci=0.95):
    """Plot multi-experiment comparison view (4 subplots: metric, win rate, loss rate, timeout rate).

    runs_data: list of tuples [(run_dir, logs), ...]
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"Comparison of {len(runs_data)} Experiments")

    subplots_config = [
        (axes[0, 0], metric, f"{metric} vs training step", metric),
        (axes[0, 1], "win_rate", "Win Rate vs training step", "win_rate"),
        (axes[1, 0], "loss_rate", "Loss Rate vs training step", "loss_rate"),
        (axes[1, 1], "timeout_rate", "Timeout Rate vs training step", "timeout_rate"),
    ]

    total_runs = len(runs_data)

    for ax, met, title, ylabel in subplots_config:
        for idx, (run_dir, logs) in enumerate(runs_data):
            color = get_color(idx, total_runs)
            display_name = get_display_name(run_dir.name)
            label = display_name
            plot_mean_ci(ax, logs, met, ci, color, label=label)

        ax.set_xlabel("checkpoint_step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # One central figure legend at the bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = min(total_runs, 4) if total_runs <= 8 else min(total_runs, 5)
    num_rows = (total_runs + ncol - 1) // ncol
    bottom_margin = min(0.30, 0.04 + 0.035 * num_rows)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncol,
        fontsize="medium",
        frameon=True,
    )
    fig.tight_layout(rect=[0, bottom_margin, 1, 0.98])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_names", nargs="*", default=None,
                        help="One or more run folder names under runs/, e.g. run1 run2 ... "
                             "If omitted, a picker window opens to choose from runs/.")
    parser.add_argument("--metric", default="avg_reward", help="Column from eval_log.csv to plot for reward graph (default: avg_reward)")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level for the band (default: 0.95)")
    parser.add_argument("--out", default=None, help="Path to save the figure instead of showing it interactively")
    args = parser.parse_args()

    if args.run_names:
        runs_data = [find_seed_logs(name) for name in args.run_names]
    else:
        runs = scan_runs()
        if not runs:
            raise FileNotFoundError(f"No runs with eval_log.csv found under {RUNS_DIR}")
        chosen_runs = select_run_gui(runs)
        if not chosen_runs:
            print("No run selected, exiting.")
            return
        runs_data = [find_seed_logs(run_path) for run_path in chosen_runs]

    print(f"Loaded {len(runs_data)} experiment(s):")
    for r_dir, logs in runs_data:
        print(f"  - {r_dir.name}: {len(logs)} seed(s)")

    if len(runs_data) == 1:
        run_dir, logs = runs_data[0]
        fig = plot_single_run(run_dir, logs, metric=args.metric, ci=args.ci)
    else:
        fig = plot_multi_runs(runs_data, metric=args.metric, ci=args.ci)

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
