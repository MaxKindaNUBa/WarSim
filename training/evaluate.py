"""Fixed-seed evaluation: rollout() reports per-policy metrics (win/loss/
timeout rates, avg steps-to-kill, avg HP remaining, hit-rate, ammo-
efficiency, empty-gun-fire attempts), and the CLI compares a trained DQN
checkpoint against the random and heuristic baselines on identical seeds.

rollout() and eval_log_row() are also imported by training/train.py for its
periodic in-loop evaluation, so both call sites share one implementation of
"what does an evaluation pass measure" and one CSV row schema.
"""
import argparse
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


def _sim_defaults():
    return _load_yaml(CONFIG_DIR / "sim_default.yaml")


def _training_defaults():
    return _load_yaml(CONFIG_DIR / "training_default.yaml")


def load_dqn_agent(env, checkpoint_path, train_config):
    agent = DQNAgent.from_config(env.observation_space.shape[0], env.action_space.n, train_config)
    agent.load_checkpoint(checkpoint_path)
    return agent


def rollout(env, act, seeds):
    """Run one episode per seed under a no-exploration policy `act(obs) -> action`."""
    outcomes = {"win": 0, "loss": 0, "timeout": 0}
    total_reward = 0.0
    win_lengths = []
    final_soldier_hp = []
    final_enemy_hp = []
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
        final_enemy_hp.append(env.enemy["hp"])
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
        "avg_enemy_hp_remaining": float(np.mean(final_enemy_hp)),
        "hit_rate": total_shots_landed / total_shots_fired if total_shots_fired else float("nan"),
        "ammo_efficiency_damage_per_shot": (
            total_damage_dealt / total_shots_fired if total_shots_fired else float("nan")
        ),
        "empty_gun_fire_attempts": empty_gun_fire_attempts,
        "shots_fired": total_shots_fired,
        "avg_shots_fired": total_shots_fired / n,
    }


EVAL_LOG_FIELDNAMES = [
    "checkpoint_step", "avg_reward", "win_rate", "loss_rate", "timeout_rate",
    "avg_steps_to_kill", "avg_soldier_hp_remaining", "avg_enemy_hp_remaining",
    "avg_shots_fired", "avg_damage_per_shot", "avg_shot_accuracy", "empty_gun_fire_attempts",
]


def eval_log_row(checkpoint_step, stats):
    """Build one eval_log.csv row from rollout() stats.

    Pure — no I/O. training/train.py hands this straight to an AsyncCSVLogger so the
    CUDA training loop never blocks on the disk write.
    """
    return {
        "checkpoint_step": checkpoint_step,
        "avg_reward": stats["avg_reward"],
        "win_rate": stats["win_rate"],
        "loss_rate": stats["loss_rate"],
        "timeout_rate": stats["timeout_rate"],
        "avg_steps_to_kill": stats["avg_steps_to_kill"],
        "avg_soldier_hp_remaining": stats["avg_soldier_hp_remaining"],
        "avg_enemy_hp_remaining": stats["avg_enemy_hp_remaining"],
        "avg_shots_fired": stats["avg_shots_fired"],
        "avg_damage_per_shot": stats["ammo_efficiency_damage_per_shot"],
        "avg_shot_accuracy": stats["hit_rate"],
        "empty_gun_fire_attempts": stats["empty_gun_fire_attempts"],
    }


def compare(checkpoint, env_config=None, reward_config=None, train_config=None,
            eval_config=None, agent_config=None):
    env_config = env_config or _sim_defaults()["env"]
    reward_config = reward_config or _training_defaults()["reward"]
    train_config = train_config or _training_defaults()["train"]
    eval_config = eval_config or _training_defaults()["eval"]
    agent_config = agent_config or _sim_defaults()["agent"]

    n_episodes = eval_config["n_episodes"]
    seed_base = eval_config["seed_base"]
    enemy_fire_enabled = eval_config["enemy_fire_enabled"]
    preferred_range = agent_config["heuristic"]["preferred_range"]

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
        "avg_steps_to_kill", "avg_soldier_hp_remaining", "avg_enemy_hp_remaining",
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
    parser.add_argument("--env-config", default=None,
                         help="Path to an env config YAML (default: env: section of config/sim_default.yaml)")
    parser.add_argument("--reward-config", default=None,
                         help="Path to a reward config YAML (default: reward: section of config/training_default.yaml)")
    parser.add_argument("--train-config", default=None,
                         help="Path to a train config YAML — must match the checkpoint's architecture "
                              "(default: train: section of config/training_default.yaml)")
    parser.add_argument("--eval-config", default=None,
                         help="Path to an eval config YAML (default: eval: section of config/training_default.yaml)")
    parser.add_argument("--agent-config", default=None,
                         help="Path to an agent config YAML for the heuristic baseline "
                              "(default: agent: section of config/sim_default.yaml)")
    parser.add_argument("--episodes", type=int, default=None,
                         help="Override eval config's n_episodes")
    parser.add_argument("--seed-base", type=int, default=None,
                         help="Override eval config's seed_base")
    args = parser.parse_args()

    eval_config = _load_yaml(args.eval_config) if args.eval_config else _training_defaults()["eval"]
    if args.episodes is not None:
        eval_config["n_episodes"] = args.episodes
    if args.seed_base is not None:
        eval_config["seed_base"] = args.seed_base

    eval_results = compare(
        args.checkpoint,
        env_config=_load_yaml(args.env_config) if args.env_config else None,
        reward_config=_load_yaml(args.reward_config) if args.reward_config else None,
        train_config=_load_yaml(args.train_config) if args.train_config else None,
        eval_config=eval_config,
        agent_config=_load_yaml(args.agent_config) if args.agent_config else None,
    )
    _print_table(eval_results)

    out_path = Path(args.checkpoint).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved detailed results to {out_path}")
