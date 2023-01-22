import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.agents.base_agent import Agent
from src.commons.memory import PrioritizedMemory, ReplayMemory
from src.commons.model import CNNModel, Model


class DQNAgent(Agent):
    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        config: dict = {},
        is_fixed_seed: bool = False,
    ) -> None:
        super().__init__(obs_space_dims, action_space_dims, config)

        # set seed number
        if is_fixed_seed:
            np.random.seed(self.config.seed)

        # set Q Network (CartPole)
        self.q_target = Model(obs_space_dims, action_space_dims).to(self.config.device)
        self.q_predict = Model(obs_space_dims, action_space_dims).to(self.config.device)

        # TODO (atari or mujoco)
        # self.q_target = CNNModel(obs_space_dims, action_space_dims)
        # self.q_predict = CNNModel(obs_space_dims, action_space_dims)

        # Memory Type
        if "priority" in self.config.memory_type:
            # prioritized
            self.memory = PrioritizedMemory(self.config.memory_capacity)
        else:
            # uniform
            self.memory = ReplayMemory(self.config.memory_capacity)

        # Optimizer
        self.optimizer = optim.Adam(self.q_predict.parameters(), lr=self.config.lr)

        # Loss function
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        # epsilon greedy exploration
        if self.epsilon > np.random.random():
            action = np.random.choice(np.arange(self.action_space_dims), 1)[0]
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state, device=self.config.device, dtype=torch.float32
                ).unsqueeze(dim=0)
                q_values = self.q_predict(state)
                action = q_values.max(1)[1].item()
        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.store(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory.buffer) < self.config.batch_size:
            return

        # Sample random minibatch of transitions
        experiences = self.memory.sample(self.config.batch_size)
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self._get_tensor_batch_from_experiences(experiences)

        q_values = self.q_predict(state_batch).gather(1, action_batch)
        next_q_values = self.q_target(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.config.gamma * next_q_values * (
            1 - done_batch
        )
        loss = self.loss_fn(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_tensor_batch_from_experiences(self, experiences):
        states = torch.tensor(
            [e.state for e in experiences if e is not None],
            device=self.config.device,
            dtype=torch.float,
        )

        actions = torch.tensor(
            [e.action for e in experiences if e is not None], device=self.config.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            [e.reward for e in experiences if e is not None],
            device=self.config.device,
            dtype=torch.float,
        )

        next_states = torch.tensor(
            [e.next_state for e in experiences if e is not None],
            device=self.config.device,
            dtype=torch.float,
        )

        dones = torch.tensor(
            [e.done for e in experiences if e is not None],
            device=self.config.device,
            dtype=torch.uint8,
        )
        return states, actions, rewards, next_states, dones
