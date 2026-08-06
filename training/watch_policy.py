"""Load a checkpoint (or a baseline policy) and watch it play in the pygame
GUI, paced to real time so playback matches timestep_duration_s.

Mechanism (action plan Section 5): record t_start at the top of each loop
iteration; after env.step() + env.render() complete, sleep for whatever's
left of the tick budget. Only ever slows down, never speeds up, so a slow
render frame just eats into the sleep budget instead of desyncing the sim.
"""
import argparse
import time
from pathlib import Path

import yaml

from agents.dqn import DQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from envs.combat_env import CombatEnv

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_dqn_agent(env, checkpoint_path, train_config):
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


def watch(policy, checkpoint=None, episodes=5, seed=0, enemy_fire_enabled=True):
    env_config = _load_yaml(CONFIG_DIR / "env_default.yaml")
    reward_config = _load_yaml(CONFIG_DIR / "reward_default.yaml")

    env = CombatEnv(
        env_config=env_config, reward_config=reward_config,
        render_mode="human", enemy_fire_enabled=enemy_fire_enabled,
    )

    if policy == "random":
        agent = RandomAgent(env.action_space)
        act = lambda obs: agent.act(obs)
    elif policy == "heuristic":
        agent = HeuristicAgent(env.n_bins, env.bin_size_degrees)
        act = lambda obs: agent.act(obs)
    elif policy == "dqn":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --policy dqn")
        train_config = _load_yaml(CONFIG_DIR / "train_default.yaml")
        agent = _load_dqn_agent(env, checkpoint, train_config)
        act = lambda obs: agent.act(obs, global_step=0, greedy=True)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    timestep_duration_s = env.timestep_duration_s

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            terminated = truncated = False
            while not (terminated or truncated):
                t_start = time.perf_counter()
                action = act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                elapsed = time.perf_counter() - t_start
                time.sleep(max(0.0, timestep_duration_s - elapsed))
            print(f"Episode {ep}: outcome={info['outcome']}")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "heuristic", "dqn"], default="heuristic")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-enemy-fire", action="store_true")
    args = parser.parse_args()
    watch(
        policy=args.policy, checkpoint=args.checkpoint, episodes=args.episodes,
        seed=args.seed, enemy_fire_enabled=not args.no_enemy_fire,
    )
