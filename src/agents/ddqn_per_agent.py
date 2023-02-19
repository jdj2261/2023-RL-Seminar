import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from src.agents.base_agent import Agent
from src.commons.memory import PrioritizedMemory


class DDQNPerAgent(Agent):
    def __init__(
        self,
        obs_space_shape: tuple,
        action_space_dims: int,
        is_atari: bool = True,
        config: dict = {},
    ) -> None:
        super().__init__(obs_space_shape, action_space_dims, config)

        self.target_network, self.policy_network = super()._get_q_models(
            obs_space_shape, action_space_dims, self.config.device, is_atari
        )
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.lr)

        self.is_atari = is_atari
        self.loss_fn = nn.SmoothL1Loss() if self.is_atari else nn.MSELoss()

        # Memory Type
        self.memory = PrioritizedMemory(self.config.buffer_size)

        # soft target update parameter
        self.tau = 1e-2

    def select_action(self, state, eps=0.0):
        state = (
            torch.from_numpy(np.array(state))
            .float()
            .unsqueeze(0)
            .to(self.config.device)
        )
        if random.random() < eps:
            return random.choice(np.arange(self.action_space_dims))
        else:
            q_values = self.policy_network(state).detach()
            _, action = q_values.max(1)
            return action.item()

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.store(state, action, reward, next_state, done)

    def update(self):
        batches, idxs, IS_weights = self.memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = batches

        states = torch.from_numpy(np.array(states)).float().to(self.config.device)
        actions = torch.from_numpy(actions).long().to(self.config.device).reshape(-1, 1)
        rewards = (
            torch.from_numpy(rewards).float().to(self.config.device).reshape(-1, 1)
        )
        next_states = (
            torch.from_numpy(np.array(next_states)).float().to(self.config.device)
        )
        dones = torch.from_numpy(dones).float().to(self.config.device).reshape(-1, 1)
        IS_weights = torch.FloatTensor(IS_weights).to(self.config.device).reshape(-1, 1)

        cur_q_values = self.policy_network(states).gather(1, actions)
        with torch.no_grad():
            _, max_next_action = self.policy_network(next_states).max(1)
            max_next_q_values = self.target_network(next_states).gather(
                1, max_next_action.unsqueeze(1)
            )
            target_q_values = rewards + (
                1 - dones
            ) * self.config.gamma * max_next_q_values.view(self.config.batch_size, -1)

        loss = torch.mean((cur_q_values - target_q_values) ** 2 * IS_weights)
        # update priority
        errors = (
            torch.abs(target_q_values - cur_q_values).detach().flatten().cpu().numpy()
        )
        for idx, error in zip(idxs, errors):
            self.memory.update_priority(idx, error)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def soft_update_target_network(self):
        for target_param, policy_param in zip(
            self.target_network.parameters(), self.policy_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
