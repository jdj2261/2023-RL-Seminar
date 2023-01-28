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
        obs_space_shape: tuple,
        action_space_dims: int,
        config: dict = {},
        is_fixed_seed: bool = False,
        is_atari: bool = True,
    ) -> None:
        super().__init__(obs_space_shape, action_space_dims, config)

        # set seed number
        if is_fixed_seed:
            np.random.seed(self.config.seed)

        if not is_atari:
            # set Q Network (CartPole)
            self.q_target = Model(obs_space_shape, action_space_dims).to(
                self.config.device
            )
            self.q_predict = Model(obs_space_shape, action_space_dims).to(
                self.config.device
            )
        else:
            # TODO (atari or mujoco)
            self.q_target = CNNModel(obs_space_shape, action_space_dims).to(
                self.config.device, non_blocking=True
            )
            self.q_predict = CNNModel(obs_space_shape, action_space_dims).to(
                self.config.device, non_blocking=True
            )

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
        self.loss = None
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss()

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
                action = torch.argmax(q_values).item()
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

        next_q_values = self.q_target(next_state_batch).max(dim=1).values.detach()
        expected_q_values = reward_batch + self.config.gamma * next_q_values * (
            1 - done_batch
        )
        current_q_values = self.q_predict(state_batch).gather(1, action_batch)
        loss = self.loss_fn(expected_q_values.unsqueeze(1), current_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

    def _get_tensor_batch_from_experiences(self, experiences):
        states = torch.tensor(
            np.array([e.state for e in experiences if e is not None]),
            device=self.config.device,
            dtype=torch.float32,
        )

        actions = torch.tensor(
            np.array([e.action for e in experiences if e is not None]),
            device=self.config.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            np.array([e.reward for e in experiences if e is not None]),
            device=self.config.device,
            dtype=torch.float32,
        )

        next_states = torch.tensor(
            np.array([e.next_state for e in experiences if e is not None]),
            device=self.config.device,
            dtype=torch.float32,
        )

        dones = torch.tensor(
            np.array([e.done for e in experiences if e is not None]),
            device=self.config.device,
            dtype=torch.uint8,
        )

        return states, actions, rewards, next_states, dones
