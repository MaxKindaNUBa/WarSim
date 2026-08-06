"""QNetwork (MLP) and DQNAgent — epsilon-greedy action selection, Double DQN
target computation, hard target-network sync, and the training step.

All network weights, replay-buffer tensors, and training math run on CUDA
(the RTX 4060) — see action plan Section 2a. Device selection fails loudly
rather than silently falling back to CPU.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_layers):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, hidden_layers, learning_rate, gamma,
                 target_sync_interval_steps, epsilon_start, epsilon_end,
                 epsilon_decay_steps, seed=None):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This project trains on the RTX 4060 "
                "by design (action plan Section 2a) — refusing to silently "
                "fall back to CPU."
            )
        self.device = torch.device("cuda")

        self.n_actions = n_actions
        self.gamma = gamma
        self.target_sync_interval_steps = target_sync_interval_steps

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.q_network = QNetwork(obs_dim, n_actions, hidden_layers).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions, hidden_layers).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self._train_steps = 0
        self._rng = random.Random(seed)

    def epsilon_at(self, global_step):
        frac = min(1.0, global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs, global_step, greedy=False):
        epsilon = 0.0 if greedy else self.epsilon_at(global_step)
        if self._rng.random() < epsilon:
            return self._rng.randrange(self.n_actions)

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch):
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

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_sync_interval_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

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
