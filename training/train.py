"""Main DQN training loop entrypoint: train() builds a CombatEnv, a DQNAgent, and
a PrioritizedReplayBuffer from config, then runs the standard off-policy loop
(act -> step -> push transition -> sample batch -> train_step) up to
train_config["num_steps_total"] env steps.

On top of that core loop this also drives several optional features, all
config-gated in config/training_default.yaml's train: section so the default
config reproduces a plain DQN+PER run:
  - enemy-difficulty curriculum (train.curriculum): starts the scripted enemy
    weak/non-firing and steps it toward full difficulty once a separate
    curriculum-check eval's win rate converges, re-injecting epsilon
    exploration on each stage advance (see agents/dqn.py:DQNAgent.bump_epsilon).
  - periodic replay-buffer top-up with fresh HeuristicAgent rollouts
    (train.replay_buffer.buffer_seeding), so "how to aim and engage"
    transitions don't get evicted out of the buffer as training goes on.
  - periodic checkpointing, both on a fixed step interval and whenever a
    greedy eval rollout beats the best win rate seen so far
    (checkpoints/checkpoint_best.pt).
  - periodic fixed-seed eval rollouts logged to eval_log.csv via a background
    thread (logging_utils/async_csv_logger.py) so the CUDA loop never blocks
    on the disk write.

Run this as `python -m training.train --run-name <name>`, or import train()
directly (training/train_multi_seed.py does this to fan a run out across
several seeds as separate processes).
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.dqn import DQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.replay_buffer import PrioritizedReplayBuffer
from envs.combat_env import CombatEnv
from logging_utils.async_csv_logger import AsyncCSVLogger
from logging_utils.run_logger import start_run
from training.evaluate import EVAL_LOG_FIELDNAMES, eval_log_row, rollout

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

logger = logging.getLogger(__name__)


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_default_configs():
    sim = _load_yaml(CONFIG_DIR / "sim_default.yaml")
    training = _load_yaml(CONFIG_DIR / "training_default.yaml")
    return sim["env"], sim["agent"], training["reward"], training["train"]


def _seed_buffer_with_heuristic(env_config, reward_config, agent_config, enemy_fire_enabled,
                                 replay_buffer, n_episodes, seed_base,
                                 enemy_reaction_interval_steps=1, enemy_fire_cooldown_steps=None):
    """Roll out HeuristicAgent for n_episodes (seeds seed_base..seed_base+n_episodes-1) on a
    scratch CombatEnv and push every transition into replay_buffer — see buffer_seeding in
    config/training_default.yaml. Returns the number of transitions pushed, for logging.

    enemy_reaction_interval_steps/enemy_fire_cooldown_steps should mirror the *current*
    curriculum stage (see train.curriculum) so seeded demonstrations match the difficulty
    the agent is actually training against, not always full difficulty.
    """
    seed_env = CombatEnv(
        env_config=env_config, reward_config=reward_config, enemy_fire_enabled=enemy_fire_enabled,
        enemy_reaction_interval_steps=enemy_reaction_interval_steps,
        enemy_fire_cooldown_steps=enemy_fire_cooldown_steps,
    )
    preferred_range = agent_config["heuristic"]["preferred_range"]
    heuristic = HeuristicAgent(seed_env.n_bins, seed_env.bin_size_degrees, preferred_range=preferred_range)

    pushed = 0
    for ep_seed in range(seed_base, seed_base + n_episodes):
        obs, info = seed_env.reset(seed=ep_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action = heuristic.act(obs)
            next_obs, reward, terminated, truncated, info = seed_env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            pushed += 1
    return pushed


def train(run_name="dqn_v1", env_config=None, agent_config=None, reward_config=None, train_config=None,
          run_dir=None):
    default_env, default_agent, default_reward, default_train = _load_default_configs()
    env_config = env_config or default_env
    agent_config = agent_config or default_agent
    reward_config = reward_config or default_reward
    train_config = train_config or default_train

    seed = train_config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir, writer = start_run(
        run_name, {"env": env_config, "agent": agent_config, "reward": reward_config, "train": train_config},
        run_dir=run_dir,
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Starting run %s in %s", run_name, run_dir)

    # Eval rows are handed to a background thread (see logging_utils/async_csv_logger.py)
    # so the disk write never blocks the CUDA training loop below.
    eval_csv_logger = AsyncCSVLogger(run_dir / "eval_log.csv", EVAL_LOG_FIELDNAMES)

    env = CombatEnv(
        env_config=env_config, reward_config=reward_config,
        enemy_fire_enabled=train_config["enemy_fire_enabled"],
    )
    eval_env = CombatEnv(
        env_config=env_config, reward_config=reward_config,
        enemy_fire_enabled=train_config["enemy_fire_enabled"],
    )
    eval_seed_base = train_config["checkpoint"].get("eval_seed_base", 900_000)
    eval_seeds = list(range(eval_seed_base, eval_seed_base + train_config["checkpoint"]["eval_episodes"]))

    # Enemy-difficulty curriculum (see train.curriculum in config/training_default.yaml).
    # `env` (training) and `curriculum_eval_env` (curriculum-check eval, below) both track the
    # current stage; `eval_env` above stays at full difficulty always, so eval_log.csv is one
    # consistent metric for the whole run regardless of what stage training is currently at.
    curriculum_config = train_config.get("curriculum", {})
    curriculum_enabled = curriculum_config.get("enabled", False) and bool(curriculum_config.get("stages"))
    curriculum_stages = curriculum_config.get("stages", [])
    curriculum_win_rate_threshold = curriculum_config.get("win_rate_threshold")
    curriculum_convergence_window = curriculum_config.get("convergence_window")
    curriculum_stage_idx = 0
    curriculum_win_rate_history = []
    curriculum_eval_env = None

    epsilon_bump_config = curriculum_config.get("epsilon_bump", {})
    epsilon_bump_enabled = epsilon_bump_config.get("enabled", False)
    epsilon_bump_value = epsilon_bump_config.get("value")  # None -> DQNAgent.bump_epsilon defaults to epsilon_start

    def _apply_curriculum_stage(stage_idx):
        stage = curriculum_stages[stage_idx]
        fire_enabled = stage.get("enemy_fire_enabled", True)
        env.set_enemy_difficulty(
            reaction_interval_steps=stage["reaction_interval_steps"],
            fire_cooldown_steps=stage["fire_cooldown_steps"],
            fire_enabled=fire_enabled,
        )
        curriculum_eval_env.set_enemy_difficulty(
            reaction_interval_steps=stage["reaction_interval_steps"],
            fire_cooldown_steps=stage["fire_cooldown_steps"],
            fire_enabled=fire_enabled,
        )
        logger.info(
            "Curriculum: enemy at stage %d/%d (enemy_fire_enabled=%s, reaction_interval_steps=%d, "
            "fire_cooldown_steps=%d)",
            stage_idx, len(curriculum_stages) - 1, fire_enabled,
            stage["reaction_interval_steps"], stage["fire_cooldown_steps"],
        )

    def _current_enemy_difficulty():
        """(reaction_interval_steps, fire_cooldown_steps, fire_enabled) for the current
        curriculum stage, or full difficulty if curriculum is off — for buffer seeding to
        mirror training."""
        if curriculum_enabled:
            stage = curriculum_stages[curriculum_stage_idx]
            return (
                stage["reaction_interval_steps"], stage["fire_cooldown_steps"],
                stage.get("enemy_fire_enabled", True),
            )
        return 1, None, train_config["enemy_fire_enabled"]

    def _maybe_advance_curriculum(curriculum_win_rate, global_step):
        nonlocal curriculum_stage_idx
        curriculum_win_rate_history.append(curriculum_win_rate)
        if curriculum_stage_idx >= len(curriculum_stages) - 1:
            return  # already at full difficulty
        recent = curriculum_win_rate_history[-curriculum_convergence_window:]
        if len(recent) >= curriculum_convergence_window and (
            sum(recent) / len(recent) >= curriculum_win_rate_threshold
        ):
            curriculum_stage_idx += 1
            curriculum_win_rate_history.clear()
            _apply_curriculum_stage(curriculum_stage_idx)
            if epsilon_bump_enabled:
                bumped = agent.bump_epsilon(global_step, value=epsilon_bump_value)
                if bumped:
                    logger.info(
                        "Epsilon bump: -> %.3f at step %d (decaying back over the existing "
                        "epsilon_decay_steps=%d)",
                        agent.epsilon_at(global_step), global_step, agent.epsilon_decay_steps,
                    )
                else:
                    logger.info(
                        "Epsilon bump: skipped at step %d (current epsilon %.3f already >= bump target)",
                        global_step, agent.epsilon_at(global_step),
                    )

    if curriculum_enabled:
        curriculum_eval_env = CombatEnv(
            env_config=env_config, reward_config=reward_config,
            enemy_fire_enabled=train_config["enemy_fire_enabled"],
        )
        _apply_curriculum_stage(curriculum_stage_idx)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent.from_config(obs_dim, n_actions, train_config, seed=seed)

    per_config = train_config["replay_buffer"].get("per", {})
    replay_buffer = PrioritizedReplayBuffer(
        capacity=train_config["replay_buffer"]["capacity"],
        obs_dim=obs_dim,
        alpha=per_config.get("alpha", 0.6),
        beta_start=per_config.get("beta_start", 0.4),
        beta_end=per_config.get("beta_end", 1.0),
        beta_decay_steps=per_config.get("beta_decay_steps", train_config["num_steps_total"]),
        priority_epsilon=per_config.get("priority_epsilon", 1e-6),
    )
    batch_size = train_config["replay_buffer"]["batch_size"]
    min_size_before_training = train_config["replay_buffer"]["min_size_before_training"]
    train_every_n_steps = train_config["train_every_n_steps"]
    log_interval_steps = train_config["log_interval_steps"]

    buffer_seeding_config = train_config["replay_buffer"].get("buffer_seeding", {})
    buffer_seeding_enabled = buffer_seeding_config.get("enabled", False)
    buffer_seeding_interval_steps = buffer_seeding_config.get("interval_steps")
    buffer_seeding_episodes_per_topup = buffer_seeding_config.get("episodes_per_topup")
    next_buffer_seed = buffer_seeding_config.get("seed_base")

    def _topup_buffer_with_heuristic():
        nonlocal next_buffer_seed
        reaction_interval_steps, fire_cooldown_steps, fire_enabled = _current_enemy_difficulty()
        pushed = _seed_buffer_with_heuristic(
            env_config, reward_config, agent_config, fire_enabled,
            replay_buffer, buffer_seeding_episodes_per_topup, next_buffer_seed,
            enemy_reaction_interval_steps=reaction_interval_steps,
            enemy_fire_cooldown_steps=fire_cooldown_steps,
        )
        logger.info(
            "Buffer top-up: pushed %d heuristic transitions from %d episodes (seeds %d..%d)",
            pushed, buffer_seeding_episodes_per_topup, next_buffer_seed,
            next_buffer_seed + buffer_seeding_episodes_per_topup - 1,
        )
        next_buffer_seed += buffer_seeding_episodes_per_topup

    if buffer_seeding_enabled:
        _topup_buffer_with_heuristic()  # initial seed, before any self-play — see buffer_seeding in config

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_interval_steps = train_config["checkpoint"]["interval_steps"]
    eval_interval_steps = train_config["checkpoint"]["eval_interval_steps"]

    global_step = 0
    best_win_rate = -1.0
    last_loss = None
    sample_rng = np.random.default_rng(seed)

    # Episode-level stats (reward/length/outcome/hit-rate/ammo) only exist once an episode
    # ends, but everything here is reported on a step cadence, not an episode cadence — so
    # completed episodes accumulate into this window and get flushed (printed + written to
    # tensorboard) the next time global_step crosses a log_interval_steps boundary, as an
    # average over however many episodes finished during that window.
    window = {
        "episodes": 0, "reward_sum": 0.0, "length_sum": 0, "wins": 0,
        "shots_fired": 0, "shots_landed": 0, "ammo_used_sum": 0,
    }

    def _flush_episode_window(step):
        if window["episodes"] == 0:
            return
        n = window["episodes"]
        avg_reward = window["reward_sum"] / n
        avg_length = window["length_sum"] / n
        win_rate = window["wins"] / n
        hit_rate = window["shots_landed"] / window["shots_fired"] if window["shots_fired"] else 0.0
        avg_ammo_used = window["ammo_used_sum"] / n
        logger.info(
            "step=%d episodes_completed=%d avg_reward=%.3f avg_length=%.1f win_rate=%.3f "
            "hit_rate=%.3f avg_ammo_used=%.1f",
            step, n, avg_reward, avg_length, win_rate, hit_rate, avg_ammo_used,
        )
        writer.add_scalar("episode/reward", avg_reward, step)
        writer.add_scalar("episode/length", avg_length, step)
        writer.add_scalar("episode/hit_rate", hit_rate, step)
        writer.add_scalar("episode/win_rate", win_rate, step)
        window["episodes"] = 0
        window["reward_sum"] = 0.0
        window["length_sum"] = 0
        window["wins"] = 0
        window["shots_fired"] = 0
        window["shots_landed"] = 0
        window["ammo_used_sum"] = 0

    num_steps_total = train_config["num_steps_total"]
    episode = 0

    try:
        while global_step < num_steps_total:
            episode += 1
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
                    states, actions, rews, next_states, ds, indices, is_weights = replay_buffer.sample(
                        batch_size, global_step=global_step, rng=sample_rng
                    )
                    last_loss, td_errors = agent.train_step(
                        (states, actions, rews, next_states, ds), weights=is_weights
                    )
                    replay_buffer.update_priorities(indices, td_errors)

                # Tensorboard/checkpoint/eval/episode-summary cadence is driven by global_step,
                # not episode count, so it means the same thing regardless of how long episodes run.
                if global_step % log_interval_steps == 0:
                    writer.add_scalar("train/epsilon", agent.epsilon_at(global_step), global_step)
                    writer.add_scalar("train/learning_rate", agent.optimizer.param_groups[0]["lr"], global_step)
                    if last_loss is not None:
                        writer.add_scalar("train/loss", last_loss, global_step)
                    _flush_episode_window(global_step)

                if buffer_seeding_enabled and global_step % buffer_seeding_interval_steps == 0:
                    _topup_buffer_with_heuristic()

                if global_step % checkpoint_interval_steps == 0:
                    agent.save_checkpoint(checkpoint_dir / f"checkpoint_step{global_step}.pt")
                    logger.info("Saved checkpoint at step %d (episode %d)", global_step, episode)

                if global_step % eval_interval_steps == 0:
                    eval_stats = rollout(
                        eval_env, lambda obs: agent.act(obs, global_step=0, greedy=True), eval_seeds
                    )
                    logger.info("Eval at step %d (episode %d): %s", global_step, episode, eval_stats)
                    eval_csv_logger.log(eval_log_row(global_step, eval_stats))
                    writer.add_scalar("eval/win_rate", eval_stats["win_rate"], global_step)
                    writer.add_scalar("eval/avg_reward", eval_stats["avg_reward"], global_step)
                    if eval_stats["win_rate"] > best_win_rate:
                        best_win_rate = eval_stats["win_rate"]
                        agent.save_checkpoint(checkpoint_dir / "checkpoint_best.pt")
                        logger.info("New best win rate %.3f at step %d", best_win_rate, global_step)

                    if curriculum_enabled:
                        curriculum_eval_stats = rollout(
                            curriculum_eval_env, lambda obs: agent.act(obs, global_step=0, greedy=True), eval_seeds
                        )
                        logger.info(
                            "Curriculum-check eval at step %d (stage %d/%d): win_rate=%.3f",
                            global_step, curriculum_stage_idx, len(curriculum_stages) - 1,
                            curriculum_eval_stats["win_rate"],
                        )
                        _maybe_advance_curriculum(curriculum_eval_stats["win_rate"], global_step)

            # Episode just ended: accumulate into the window only — the actual print/log
            # happens on the next log_interval_steps boundary, not here (see _flush_episode_window).
            window["episodes"] += 1
            window["reward_sum"] += ep_reward
            window["length_sum"] += ep_length
            window["wins"] += int(info["outcome"] == "win")
            window["shots_fired"] += shots_fired
            window["shots_landed"] += shots_landed
            window["ammo_used_sum"] += ammo_used

        _flush_episode_window(global_step)  # flush whatever's left of a partial window

        final_summary = {
            "run_name": run_name,
            "num_steps_total": num_steps_total,
            "global_steps": global_step,
            "num_episodes_completed": episode,
            "best_eval_win_rate": best_win_rate,
        }
        with open(run_dir / "final_summary.json", "w") as f:
            json.dump(final_summary, f, indent=2)
        logger.info("Training complete: %s", final_summary)

    except Exception:
        logger.exception("Training loop crashed")
        raise
    finally:
        eval_csv_logger.close()
        writer.close()

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="dqn_v3_PER_curriculum")
    parser.add_argument("--env-config", default=None,
                         help="Path to an env config YAML (default: env: section of config/sim_default.yaml)")
    parser.add_argument("--agent-config", default=None,
                         help="Path to an agent config YAML for the heuristic buffer-seeding policy "
                              "(default: agent: section of config/sim_default.yaml)")
    parser.add_argument("--reward-config", default=None,
                         help="Path to a reward config YAML (default: reward: section of config/training_default.yaml)")
    parser.add_argument("--train-config", default=None,
                         help="Path to a train config YAML (default: train: section of config/training_default.yaml)")
    args = parser.parse_args()
    train(
        run_name=args.run_name,
        env_config=_load_yaml(args.env_config) if args.env_config else None,
        agent_config=_load_yaml(args.agent_config) if args.agent_config else None,
        reward_config=_load_yaml(args.reward_config) if args.reward_config else None,
        train_config=_load_yaml(args.train_config) if args.train_config else None,
    )
