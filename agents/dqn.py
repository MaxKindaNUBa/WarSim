"""QNetwork (MLP) and DQNAgent — epsilon-greedy action selection, Double DQN
target computation, target-network update (hard sync or Polyak averaging), and
the training step.

All network weights, replay-buffer tensors, and training math run on CUDA (this
project trains on an RTX 4060 by design). Device selection fails loudly rather
than silently falling back to CPU: DQNAgent.__init__ raises immediately if
torch.cuda.is_available() is False.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

_LOSS_FNS = {
    "smooth_l1": F.smooth_l1_loss,
    "mse": F.mse_loss,
}


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_layers, activation="relu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}', expected one of {list(_ACTIVATIONS)}")
        activation_cls = _ACTIVATIONS[activation]
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, hidden_layers, learning_rate, gamma,
                 target_sync_interval_steps, epsilon_start, epsilon_end,
                 epsilon_decay_steps, activation="relu", optimizer_type="adam",
                 weight_decay=0.0, eps=1e-8, momentum=0.9, grad_clip_norm=None,
                 loss_fn="smooth_l1", target_update_mode="polyak", target_tau=0.005,
                 lr_decay_enabled=False, lr_end=None, lr_decay_steps=None,
                 seed=None):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This project trains on a CUDA GPU "
                "by design — refusing to silently fall back to CPU."
            )
        if loss_fn not in _LOSS_FNS:
            raise ValueError(f"Unknown loss_fn '{loss_fn}', expected one of {list(_LOSS_FNS)}")
        if target_update_mode not in ("hard", "polyak"):
            raise ValueError(
                f"Unknown target_update_mode '{target_update_mode}', expected 'hard' or 'polyak'"
            )
        self.device = torch.device("cuda")

        self.n_actions = n_actions
        self.gamma = gamma
        self.target_sync_interval_steps = target_sync_interval_steps
        self.target_update_mode = target_update_mode
        self.target_tau = target_tau
        self.grad_clip_norm = grad_clip_norm
        self.loss_fn = _LOSS_FNS[loss_fn]

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        # anchor: epsilon_at ramps from _epsilon_anchor_value at _epsilon_anchor_step toward
        # epsilon_end over epsilon_decay_steps. Both start out matching a plain linear decay
        # from step 0; bump_epsilon() moves the anchor (see train.curriculum.epsilon_bump)
        # so curriculum stage changes can re-inject exploration without restarting the agent.
        self._epsilon_anchor_step = 0
        self._epsilon_anchor_value = epsilon_start

        # Linear LR decay over train_step() count (gradient updates, not env steps) — see
        # train.optim.lr_decay in config/training_default.yaml. Off by default (lr_at just
        # returns learning_rate unchanged), matching every optimizer setting before this.
        self.lr_start = learning_rate
        self.lr_decay_enabled = lr_decay_enabled
        self.lr_end = lr_end if lr_end is not None else learning_rate
        self.lr_decay_steps = lr_decay_steps

        self.q_network = QNetwork(obs_dim, n_actions, hidden_layers, activation=activation).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions, hidden_layers, activation=activation).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = self._build_optimizer(
            optimizer_type, learning_rate, weight_decay, eps, momentum
        )

        self._train_steps = 0
        self._rng = random.Random(seed)

    def _build_optimizer(self, optimizer_type, learning_rate, weight_decay, eps, momentum):
        params = self.q_network.parameters()
        if optimizer_type == "adam":
            return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, eps=eps)
        if optimizer_type == "sgd":
            return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        if optimizer_type == "rmsprop":
            return torch.optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay, eps=eps, momentum=momentum
            )
        raise ValueError(f"Unknown optimizer_type '{optimizer_type}', expected one of: adam, sgd, rmsprop")

    @classmethod
    def from_config(cls, obs_dim, n_actions, train_config, seed=None):
        """Build a DQNAgent from a train_config dict (the `train:` section of config/training_default.yaml).

        Shared by training/train.py, training/evaluate.py, and training/watch_policy.py
        so a checkpoint's architecture/optimizer settings only need to be assembled once.
        """
        network = train_config["network"]
        optim_cfg = train_config["optim"]
        lr_decay_cfg = optim_cfg.get("lr_decay", {})
        return cls(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_layers=network["hidden_layers"],
            activation=network.get("activation", "relu"),
            learning_rate=optim_cfg["learning_rate"],
            gamma=optim_cfg["gamma"],
            optimizer_type=optim_cfg.get("optimizer_type", "adam"),
            weight_decay=optim_cfg.get("weight_decay", 0.0),
            eps=optim_cfg.get("eps", 1e-8),
            momentum=optim_cfg.get("momentum", 0.9),
            grad_clip_norm=optim_cfg.get("grad_clip_norm"),
            loss_fn=optim_cfg.get("loss_fn", "smooth_l1"),
            target_update_mode=train_config["target_network"].get("update_mode", "polyak"),
            target_tau=train_config["target_network"].get("tau", 0.005),
            target_sync_interval_steps=train_config["target_network"].get("sync_interval_steps", 1000),
            epsilon_start=train_config["epsilon"]["start"],
            epsilon_end=train_config["epsilon"]["end"],
            epsilon_decay_steps=train_config["epsilon"]["decay_steps"],
            lr_decay_enabled=lr_decay_cfg.get("enabled", False),
            lr_end=lr_decay_cfg.get("end_learning_rate"),
            lr_decay_steps=lr_decay_cfg.get("decay_steps"),
            seed=seed,
        )

    def lr_at(self, train_step):
        if not self.lr_decay_enabled:
            return self.lr_start
        frac = min(1.0, train_step / max(1, self.lr_decay_steps))
        return self.lr_start + frac * (self.lr_end - self.lr_start)

    def epsilon_at(self, global_step):
        effective_step = max(0, global_step - self._epsilon_anchor_step)
        frac = min(1.0, effective_step / max(1, self.epsilon_decay_steps))
        return self._epsilon_anchor_value + frac * (self.epsilon_end - self._epsilon_anchor_value)

    def bump_epsilon(self, global_step, value=None):
        """Re-inject exploration by resetting the decay anchor here — e.g. on a curriculum
        stage change (train.curriculum.epsilon_bump), so a policy that's gone mostly-greedy
        against the old dynamics gets a chance to re-explore the new ones, instead of quietly
        exploiting a policy that's stale for the environment it's now actually in.

        value: epsilon jumps to this immediately (default: epsilon_start, a full reset). Ramps
        back down over the existing epsilon_decay_steps, unchanged — a bump only ever adds
        exploration on top of the current schedule, it doesn't get its own separate decay rate
        (that previously let an early bump *shorten* the remaining decay budget below what the
        original schedule already had left, net-reducing total exploration instead of adding to it).

        No-op if the natural (un-bumped) epsilon at global_step is already >= the target value —
        a "bump" should only ever raise exploration, never lower it (e.g. curriculum advancing
        early, while the initial cold-start decay hasn't dropped below the bump target yet).
        Returns True if the bump was applied, False if skipped for that reason.
        """
        target_value = self.epsilon_start if value is None else value
        if target_value <= self.epsilon_at(global_step):
            return False
        self._epsilon_anchor_step = global_step
        self._epsilon_anchor_value = target_value
        return True
        return True

    def act(self, obs, global_step, greedy=False):
        epsilon = 0.0 if greedy else self.epsilon_at(global_step)
        if self._rng.random() < epsilon:
            return self._rng.randrange(self.n_actions)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch, weights=None):
        """weights: optional per-sample importance-sampling weights (PrioritizedReplayBuffer).
        None means uniform (plain mean loss, matches pre-PER behavior exactly).

        Returns (loss, td_errors) — td_errors are |q_value - target| per sample, for
        PrioritizedReplayBuffer.update_priorities(). Callers on a plain ReplayBuffer can
        just ignore the second element.
        """
        states, actions, rewards, next_states, dones = batch

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: select next action with the online network, evaluate with the target network
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        td_errors = (q_values - targets).detach().abs()
        elementwise_loss = self.loss_fn(q_values, targets, reduction="none")
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            loss = (elementwise_loss * weights).mean()
        else:
            loss = elementwise_loss.mean()

        if self.lr_decay_enabled:
            current_lr = self.lr_at(self._train_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = current_lr

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self._train_steps += 1
        self._update_target_network()

        return loss.item(), td_errors.cpu().numpy()

    def _update_target_network(self):
        if self.target_update_mode == "hard":
            if self._train_steps % self.target_sync_interval_steps == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
        else:  # polyak: every train_step, target <- tau*online + (1-tau)*target
            tau = self.target_tau
            with torch.no_grad():
                for target_param, online_param in zip(
                    self.target_network.parameters(), self.q_network.parameters()
                ):
                    target_param.mul_(1.0 - tau).add_(online_param, alpha=tau)

    def save_checkpoint(self, path):
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self._train_steps,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._train_steps = ckpt["train_steps"]
