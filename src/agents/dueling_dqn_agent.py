import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from src.agents.base_agent import Agent
from src.commons.memory import ReplayMemory


class DuelingDQNAgent(Agent):
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
        self.memory = ReplayMemory(self.config.buffer_size)
        self.tau = 1e-2

        self.agent_name = DuelingDQNAgent.__name__

    def select_action(self, state, eps=0.0):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.config.device)
        if random.random() < eps:
            return random.choice(np.arange(self.action_space_dims))
        else:
            q_values = self.policy_network(state).detach()
            _, action = q_values.max(1)
            return action.item()

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.store(state, action, reward, next_state, done)

    def update(self):
        pass

    def soft_update_target_network(self):
        for target_param, policy_param in zip(
            self.target_network.parameters(), self.policy_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
