"""Fixed-seed evaluation: rollout() reports per-policy metrics (win/loss/
timeout rates, avg steps-to-kill, avg HP remaining, hit-rate, ammo-
efficiency, empty-gun-fire attempts), and the CLI compares a trained DQN
checkpoint against the random and heuristic baselines on identical seeds.

rollout() and append_eval_log_row() are also imported by training/train.py
for its periodic in-loop evaluation, so both call sites share one
implementation of "what does an evaluation pass measure."
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from agents.dqn import DQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from envs.combat_env import CombatEnv

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dqn_agent(env, checkpoint_path, train_config):
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_layers=train_config["network"]["hidden_layers"],
        learning_rate=train_config["optim"]["learning_rate"],
        gamma=train_config["optim"]["gamma"],
        target_sync_interval_steps=train_config["target_network"]["sync_interval_steps"],
        epsilon_start=train_config["epsilon"]["start"],
        epsilon_end=train_config["epsilon"]["end"],
        epsilon_decay_steps=train_config["epsilon"]["decay_steps"],
    )
    agent.load_checkpoint(checkpoint_path)
    return agent


def rollout(env, act, seeds):
    """Run one episode per seed under a no-exploration policy `act(obs) -> action`."""
    outcomes = {"win": 0, "loss": 0, "timeout": 0}
    total_reward = 0.0
    win_lengths = []
    final_soldier_hp = []
    total_shots_fired = 0
    total_shots_landed = 0
    total_damage_dealt = 0.0
    empty_gun_fire_attempts = 0

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        ep_length = 0
        while not (terminated or truncated):
            action = act(obs)
            prev_ammo = env.soldier["ammo"]
            prev_enemy_hp = env.enemy["hp"]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_length += 1
            total_reward += reward

            if info["soldier_empty_gun_fire"]:
                empty_gun_fire_attempts += 1
            if prev_ammo > env.soldier["ammo"]:
                total_shots_fired += 1
                total_damage_dealt += prev_enemy_hp - env.enemy["hp"]
                if info["soldier_hit"]:
                    total_shots_landed += 1

        outcomes[info["outcome"]] += 1
        final_soldier_hp.append(env.soldier["hp"])
        if info["outcome"] == "win":
            win_lengths.append(ep_length)

    n = len(seeds)
    return {
        "episodes": n,
        "total_reward": total_reward,
        "avg_reward": total_reward / n,
        "win_rate": outcomes["win"] / n,
        "loss_rate": outcomes["loss"] / n,
        "timeout_rate": outcomes["timeout"] / n,
        "avg_steps_to_kill": float(np.mean(win_lengths)) if win_lengths else float("nan"),
        "avg_soldier_hp_remaining": float(np.mean(final_soldier_hp)),
        "hit_rate": total_shots_landed / total_shots_fired if total_shots_fired else float("nan"),
        "ammo_efficiency_damage_per_shot": (
            total_damage_dealt / total_shots_fired if total_shots_fired else float("nan")
        ),
        "empty_gun_fire_attempts": empty_gun_fire_attempts,
        "shots_fired": total_shots_fired,
    }


def append_eval_log_row(run_dir, checkpoint_step, stats):
    """Append one row to runs/<run>/eval_log.csv per Section 6's column spec."""
    path = Path(run_dir) / "eval_log.csv"
    fieldnames = [
        "checkpoint_step", "total_reward", "win_rate", "loss_rate", "timeout_rate",
        "avg_steps_to_kill", "avg_soldier_hp_remaining",
    ]
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "checkpoint_step": checkpoint_step,
            "total_reward": stats["total_reward"],
            "win_rate": stats["win_rate"],
            "loss_rate": stats["loss_rate"],
            "timeout_rate": stats["timeout_rate"],
            "avg_steps_to_kill": stats["avg_steps_to_kill"],
            "avg_soldier_hp_remaining": stats["avg_soldier_hp_remaining"],
        })


def compare(checkpoint, n_episodes=200, seed_base=500_000, enemy_fire_enabled=True,
            preferred_range=15.0):
    env_config = _load_yaml(CONFIG_DIR / "env_default.yaml")
    reward_config = _load_yaml(CONFIG_DIR / "reward_default.yaml")
    train_config = _load_yaml(CONFIG_DIR / "train_default.yaml")

    seeds = list(range(seed_base, seed_base + n_episodes))
    results = {}

    env = CombatEnv(env_config=env_config, reward_config=reward_config, enemy_fire_enabled=enemy_fire_enabled)
    random_agent = RandomAgent(env.action_space)
    results["random"] = rollout(env, random_agent.act, seeds)

    env = CombatEnv(env_config=env_config, reward_config=reward_config, enemy_fire_enabled=enemy_fire_enabled)
    heuristic_agent = HeuristicAgent(env.n_bins, env.bin_size_degrees, preferred_range=preferred_range)
    results["heuristic"] = rollout(env, heuristic_agent.act, seeds)

    env = CombatEnv(env_config=env_config, reward_config=reward_config, enemy_fire_enabled=enemy_fire_enabled)
    dqn_agent = load_dqn_agent(env, checkpoint, train_config)
    results["dqn"] = rollout(env, lambda obs: dqn_agent.act(obs, global_step=0, greedy=True), seeds)

    return results


def _print_table(results):
    metrics = [
        "win_rate", "loss_rate", "timeout_rate", "avg_reward", "hit_rate",
        "avg_steps_to_kill", "avg_soldier_hp_remaining",
        "ammo_efficiency_damage_per_shot", "empty_gun_fire_attempts",
    ]
    policies = list(results.keys())
    print(f"{'metric':<32}" + "".join(f"{name:>12}" for name in policies))
    for metric in metrics:
        row = f"{metric:<32}"
        for name in policies:
            value = results[name][metric]
            row += f"{value:>12.3f}" if isinstance(value, float) else f"{value:>12}"
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=500_000)
    args = parser.parse_args()

    eval_results = compare(args.checkpoint, n_episodes=args.episodes, seed_base=args.seed_base)
    _print_table(eval_results)

    out_path = Path(args.checkpoint).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")
