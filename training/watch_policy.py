"""Load a checkpoint (or a baseline policy) and watch it play in the pygame
GUI, paced to real time so playback matches timestep_duration_s.

Mechanism: record t_start at the top of each loop iteration; after
env.step() + env.render() complete, sleep for whatever's left of the tick
budget. Only ever slows down, never speeds up, so a slow render frame just
eats into the sleep budget instead of desyncing the sim.
"""

# python -m training.watch_policy --policy dqn --checkpoint runs/dqn_v1_20260806_233013/checkpoints/checkpoint_best.pt
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


def _sim_defaults():
    return _load_yaml(CONFIG_DIR / "sim_default.yaml")


def _training_defaults():
    return _load_yaml(CONFIG_DIR / "training_default.yaml")


def watch(policy, checkpoint=None, episodes=5, seed=0, enemy_fire_enabled=True,
          env_config=None, reward_config=None, render_config=None,
          train_config=None, agent_config=None):
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
    args = parser.parse_args()
    watch(
        policy=args.policy, checkpoint=args.checkpoint, episodes=args.episodes,
        seed=args.seed, enemy_fire_enabled=not args.no_enemy_fire,
        env_config=_load_yaml(args.env_config) if args.env_config else None,
        reward_config=_load_yaml(args.reward_config) if args.reward_config else None,
        render_config=_load_yaml(args.render_config) if args.render_config else None,
        train_config=_load_yaml(args.train_config) if args.train_config else None,
        agent_config=_load_yaml(args.agent_config) if args.agent_config else None,
    )
