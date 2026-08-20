"""Launch several independent DQN training runs (one per seed) concurrently.

Per the "On training speed" note in the README, a single run is CPU-bound
(single-threaded env stepping) far more than GPU-bound (128-sample batches are
too small to hide per-kernel dispatch overhead), so one run alone only occupies
a small slice of both the GPU and the CPU. Running N seeds as separate OS
processes lets their CPU-heavy env loops overlap across cores while their CUDA
calls interleave on the GPU, raising utilization without changing the
per-seed training code at all.

Each seed runs training.train.train() unmodified except for train_config["seed"]
and an explicit run_dir override, so every seed's full output (checkpoints,
eval_log.csv, run.log, tensorboard, config_snapshot.yaml, final_summary.json)
lands in its own folder nested under one shared experiment folder:

    runs/<run-name>_<timestamp>/
        <seed>/
            checkpoints/  eval_log.csv  run.log  tensorboard/  config_snapshot.yaml  final_summary.json
        <seed>/
            ...

One timestamp for the whole experiment (not one per seed) is what makes this a
single comparable batch rather than N unrelated runs scattered across runs/.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_seeds(spec):
    """'1,2,3' -> [1,2,3]; '0-4' -> [0,1,2,3,4]; ranges and singles can be mixed/repeated."""
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _train_one_seed(seed, run_name, env_config, reward_config, train_config, run_dir):
    # training.train is imported inside the worker, not at module scope: this
    # runs in a spawned child process, and CUDA needs to initialize fresh there
    # rather than being inherited from a forked parent (spawn re-runs imports
    # from scratch anyway, but keeping torch out of the parent's import graph
    # for this script keeps the launcher itself usable without a GPU).
    from training.train import train

    seed_train_config = copy.deepcopy(train_config)
    seed_train_config["seed"] = seed
    try:
        run_dir = train(
            run_name=f"{run_name}_seed{seed}",
            env_config=env_config,
            reward_config=reward_config,
            train_config=seed_train_config,
            run_dir=run_dir,
        )
        summary_path = Path(run_dir) / "final_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        return {"seed": seed, "run_dir": str(run_dir), "ok": True, "summary": summary}
    except Exception as exc:
        return {
            "seed": seed,
            "run_dir": None,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def train_multi_seed(seeds, run_name="dqn_v2_PER", max_parallel=None,
                      env_config=None, reward_config=None, train_config=None):
    if train_config is None:
        train_config = _load_yaml(CONFIG_DIR / "training_default.yaml")["train"]
    if env_config is None:
        env_config = _load_yaml(CONFIG_DIR / "sim_default.yaml")["env"]
    if reward_config is None:
        reward_config = _load_yaml(CONFIG_DIR / "training_default.yaml")["reward"]

    if max_parallel is None:
        # Leave a couple cores free for the OS/desktop; env stepping is the
        # CPU-bound part, so core count -- not GPU VRAM -- is the binding
        # constraint on this machine (see README's "On training speed").
        max_parallel = max(1, min(len(seeds), (os.cpu_count() or 4) - 2))

    # One timestamp for the whole experiment, not one per seed -- see module docstring
    # for the resulting runs/<run-name>_<timestamp>/<seed>/ layout.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = RUNS_DIR / f"{run_name}_{timestamp}"

    print(f"Launching {len(seeds)} seed(s) {seeds}, up to {max_parallel} concurrently...")
    print(f"Experiment folder: {experiment_dir}")

    results = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_parallel, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _train_one_seed, seed, run_name, env_config, reward_config, train_config,
                experiment_dir / str(seed),
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            result = future.result()
            results.append(result)
            if result["ok"]:
                win_rate = result["summary"].get("best_eval_win_rate")
                print(f"[seed {seed}] done -> {result['run_dir']} (best_eval_win_rate={win_rate})")
            else:
                print(f"[seed {seed}] FAILED: {result['error']}\n{result['traceback']}")

    results.sort(key=lambda r: r["seed"])
    print("\n=== Summary ===")
    for r in results:
        if r["ok"]:
            win_rate = r["summary"].get("best_eval_win_rate")
            print(f"seed={r['seed']:<6} best_eval_win_rate={win_rate} run_dir={r['run_dir']}")
        else:
            print(f"seed={r['seed']:<6} FAILED: {r['error']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the same config across multiple seeds concurrently, "
                     "one OS process per seed, sharing the GPU."
    )
    parser.add_argument("--seeds", default="324,235,34215,34562,45656,2467,536735,78356,2457,567,7529,25687,1457653,25671658,41567257,14574",
                         help="Comma-separated seeds and/or ranges, e.g. '1,2,3' or '0-4' or '1,5-7,20'")
    parser.add_argument("--run-name", default="dqn_v6_vdbe_sigma_15",)
    parser.add_argument("--max-parallel", type=int, default=3,
                         help="Max concurrent training processes (default: cpu_count - 2)")
    parser.add_argument("--env-config", default=None,
                         help="Path to an env config YAML (default: env: section of config/sim_default.yaml)")
    parser.add_argument("--reward-config", default=None,
                         help="Path to a reward config YAML (default: reward: section of config/training_default.yaml)")
    parser.add_argument("--train-config", default=None,
                         help="Path to a train config YAML (default: train: section of config/training_default.yaml)")
    args = parser.parse_args()

    train_multi_seed(
        seeds=_parse_seeds(args.seeds),
        run_name=args.run_name,
        max_parallel=args.max_parallel,
        env_config=_load_yaml(args.env_config) if args.env_config else None,
        reward_config=_load_yaml(args.reward_config) if args.reward_config else None,
        train_config=_load_yaml(args.train_config) if args.train_config else None,
    )
