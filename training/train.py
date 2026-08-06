"""Main DQN training loop entrypoint."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from envs.combat_env import CombatEnv
from logging_utils.episode_logger import EpisodeLogger
from logging_utils.run_logger import start_run
from training.evaluate import append_eval_log_row, rollout

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

logger = logging.getLogger(__name__)


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_default_configs():
    return (
        _load_yaml(CONFIG_DIR / "env_default.yaml"),
        _load_yaml(CONFIG_DIR / "reward_default.yaml"),
        _load_yaml(CONFIG_DIR / "train_default.yaml"),
    )


def train(run_name="dqn_v1", env_config=None, reward_config=None, train_config=None):
    default_env, default_reward, default_train = _load_default_configs()
    env_config = env_config or default_env
    reward_config = reward_config or default_reward
    train_config = train_config or default_train

    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir, writer = start_run(
        run_name, {"env": env_config, "reward": reward_config, "train": train_config}
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Starting run %s in %s", run_name, run_dir)

    episode_logger = EpisodeLogger(run_dir)

    env = CombatEnv(
        env_config=env_config, reward_config=reward_config,
        enemy_fire_enabled=train_config["enemy_fire_enabled"],
    )
    eval_env = CombatEnv(
        env_config=env_config, reward_config=reward_config,
        enemy_fire_enabled=train_config["enemy_fire_enabled"],
    )
    eval_seeds = list(range(900_000, 900_000 + train_config["checkpoint"]["eval_episodes"]))

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_layers=train_config["network"]["hidden_layers"],
        learning_rate=train_config["optim"]["learning_rate"],
        gamma=train_config["optim"]["gamma"],
        target_sync_interval_steps=train_config["target_network"]["sync_interval_steps"],
        epsilon_start=train_config["epsilon"]["start"],
        epsilon_end=train_config["epsilon"]["end"],
        epsilon_decay_steps=train_config["epsilon"]["decay_steps"],
        seed=seed,
    )

    replay_buffer = ReplayBuffer(capacity=train_config["replay_buffer"]["capacity"], obs_dim=obs_dim)
    batch_size = train_config["replay_buffer"]["batch_size"]
    min_size_before_training = train_config["replay_buffer"]["min_size_before_training"]
    train_every_n_steps = train_config["train_every_n_steps"]

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_interval_episodes = train_config["checkpoint"]["interval_episodes"]
    eval_interval_episodes = train_config["checkpoint"]["eval_interval_episodes"]

    global_step = 0
    best_win_rate = -1.0
    sample_rng = np.random.default_rng(seed)

    try:
        for episode in range(1, train_config["num_episodes"] + 1):
            obs, info = env.reset(seed=seed + episode)
            ep_reward = 0.0
            ep_length = 0
            shots_fired = 0
            shots_landed = 0
            ammo_used = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = agent.act(obs, global_step=global_step)
                prev_ammo = env.soldier["ammo"]
                next_obs, reward, terminated, truncated, info = env.step(action)

                replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)

                if prev_ammo > env.soldier["ammo"]:
                    shots_fired += 1
                    ammo_used += 1
                    if info["soldier_hit"]:
                        shots_landed += 1

                obs = next_obs
                ep_reward += reward
                ep_length += 1
                global_step += 1

                if len(replay_buffer) >= min_size_before_training and global_step % train_every_n_steps == 0:
                    batch = replay_buffer.sample(batch_size, rng=sample_rng)
                    loss = agent.train_step(batch)
                    writer.add_scalar("train/loss", loss, global_step)

                writer.add_scalar("train/epsilon", agent.epsilon_at(global_step), global_step)

            hit_rate = shots_landed / shots_fired if shots_fired else 0.0
            episode_logger.log(
                episode=episode, total_reward=ep_reward, length=ep_length,
                outcome=info["outcome"], hit_rate=hit_rate, ammo_used=ammo_used,
            )
            writer.add_scalar("episode/reward", ep_reward, episode)
            writer.add_scalar("episode/length", ep_length, episode)
            writer.add_scalar("episode/hit_rate", hit_rate, episode)
            writer.add_scalar("episode/outcome_win", float(info["outcome"] == "win"), episode)

            if episode % checkpoint_interval_episodes == 0:
                agent.save_checkpoint(checkpoint_dir / f"checkpoint_ep{episode}.pt")
                logger.info("Saved checkpoint at episode %d", episode)

            if episode % eval_interval_episodes == 0:
                eval_stats = rollout(
                    eval_env, lambda obs: agent.act(obs, global_step=0, greedy=True), eval_seeds
                )
                logger.info("Eval at episode %d: %s", episode, eval_stats)
                append_eval_log_row(run_dir, checkpoint_step=global_step, stats=eval_stats)
                writer.add_scalar("eval/win_rate", eval_stats["win_rate"], episode)
                writer.add_scalar("eval/avg_reward", eval_stats["avg_reward"], episode)
                if eval_stats["win_rate"] > best_win_rate:
                    best_win_rate = eval_stats["win_rate"]
                    agent.save_checkpoint(checkpoint_dir / "checkpoint_best.pt")
                    logger.info("New best win rate %.3f at episode %d", best_win_rate, episode)

        final_summary = {
            "run_name": run_name,
            "num_episodes": train_config["num_episodes"],
            "global_steps": global_step,
            "best_eval_win_rate": best_win_rate,
        }
        with open(run_dir / "final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)
        logger.info("Training complete: %s", final_summary)

    except Exception:
        logger.exception("Training loop crashed")
        raise
    finally:
        writer.close()

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="dqn_v1")
    args = parser.parse_args()
    train(run_name=args.run_name)
