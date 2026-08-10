"""Standalone heuristic-baseline runner. Rolls out HeuristicAgent for a batch of
fixed-seed episodes under the current env/reward config and prints the aggregate
outcome to the terminal. No files are written — use training/evaluate.py --checkpoint
if you want a saved comparison against a trained DQN checkpoint instead.
"""
import argparse
from pathlib import Path

import yaml

from agents.heuristic_agent import HeuristicAgent
from envs.combat_env import CombatEnv
from training.evaluate import rollout

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _sim_defaults():
    return _load_yaml(CONFIG_DIR / "sim_default.yaml")


def _training_defaults():
    return _load_yaml(CONFIG_DIR / "training_default.yaml")


def run_heuristic(n_episodes=500, env_config=None, reward_config=None,
                   agent_config=None, seed_base=None, enemy_fire_enabled=None):
    env_config = env_config or _sim_defaults()["env"]
    reward_config = reward_config or _training_defaults()["reward"]
    agent_config = agent_config or _sim_defaults()["agent"]

    eval_defaults = _training_defaults()["eval"]
    seed_base = seed_base if seed_base is not None else eval_defaults["seed_base"]
    enemy_fire_enabled = (
        enemy_fire_enabled if enemy_fire_enabled is not None else eval_defaults["enemy_fire_enabled"]
    )
    preferred_range = agent_config["heuristic"]["preferred_range"]

    env = CombatEnv(env_config=env_config, reward_config=reward_config,
                     enemy_fire_enabled=enemy_fire_enabled)
    agent = HeuristicAgent(env.n_bins, env.bin_size_degrees, preferred_range=preferred_range)

    seeds = list(range(seed_base, seed_base + n_episodes))
    stats = rollout(env, agent.act, seeds)

    print(f"Heuristic baseline — {n_episodes} episodes, seeds {seed_base}..{seed_base + n_episodes - 1}, "
          f"enemy_fire_enabled={enemy_fire_enabled}")
    for key, value in stats.items():
        print(f"  {key:<28} {value}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed-base", type=int, default=None,
                         help="First seed (default: eval.seed_base in config/training_default.yaml)")
    parser.add_argument("--env-config", default=None,
                         help="Path to an env config YAML (default: env: section of config/sim_default.yaml)")
    parser.add_argument("--reward-config", default=None,
                         help="Path to a reward config YAML (default: reward: section of config/training_default.yaml)")
    parser.add_argument("--agent-config", default=None,
                         help="Path to an agent config YAML for the heuristic policy "
                              "(default: agent: section of config/sim_default.yaml)")
    parser.add_argument("--no-enemy-fire", action="store_true")
    parser.add_argument("--enemy-fire", action="store_true")
    args = parser.parse_args()
    if args.no_enemy_fire and args.enemy_fire:
        parser.error("--no-enemy-fire and --enemy-fire are mutually exclusive")

    enemy_fire_enabled = None
    if args.no_enemy_fire:
        enemy_fire_enabled = False
    elif args.enemy_fire:
        enemy_fire_enabled = True

    run_heuristic(
        n_episodes=args.episodes,
        env_config=_load_yaml(args.env_config) if args.env_config else None,
        reward_config=_load_yaml(args.reward_config) if args.reward_config else None,
        agent_config=_load_yaml(args.agent_config) if args.agent_config else None,
        seed_base=args.seed_base,
        enemy_fire_enabled=enemy_fire_enabled,
    )
