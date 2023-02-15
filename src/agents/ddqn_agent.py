import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from src.agents.base_agent import Agent
from src.commons.memory import ReplayMemory
from src.commons.model import CNNModel, Model


class DDQNAgent(Agent):
    def __init__(
        self,
        obs_space_shape: tuple,
        action_space_dims: int,
        config: dict = {},
        is_fixed_seed: bool = True,
        is_atari: bool = True,
    ) -> None:
        super().__init__(obs_space_shape, action_space_dims, config)

        print("This agent is DDQN Agent!!")
        if is_fixed_seed:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
        self.is_atari = is_atari

        self.target_network, self.policy_network = self._get_q_models(
            obs_space_shape, action_space_dims, self.config.device, is_atari
        )
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.lr)

        if self.is_atari:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        self.memory = ReplayMemory(self.config.buffer_size)
        self.tau = 1e-3

    def _get_q_models(self, obs_space_shape, action_space_dims, device, is_atari):
        if not is_atari:
            target_network = Model(obs_space_shape, action_space_dims).to(device)
            policy_network = Model(obs_space_shape, action_space_dims).to(device)
        else:
            target_network = CNNModel(obs_space_shape, action_space_dims).to(
                device, non_blocking=True
            )
            policy_network = CNNModel(obs_space_shape, action_space_dims).to(
                device, non_blocking=True
            )

        return target_network, policy_network

    def select_action(self, state: np.ndarray, eps=0.0):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        if self.is_atari:
            state = np.array(state) / 255.0
        else:
            state = np.array(state)

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.config.device)
        if random.random() < eps:
            return random.choice(np.arange(self.action_space_dims))
        else:
            with torch.no_grad():
                q_values = self.policy_network(state)
                _, action = q_values.max(1)
                return action.item()

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.store(state, action, reward, next_state, done)

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )

        if self.is_atari:
            states = np.array(states) / 255.0
            next_states = np.array(next_states) / 255.0
        else:
            states = np.array(states)
            next_states = np.array(next_states)

        states = torch.from_numpy(states).float().to(self.config.device)
        actions = torch.from_numpy(actions).long().to(self.config.device)
        rewards = torch.from_numpy(rewards).float().to(self.config.device)
        next_states = torch.from_numpy(next_states).float().to(self.config.device)
        dones = torch.from_numpy(dones).float().to(self.config.device)

        with torch.no_grad():
            _, max_next_action = self.policy_network(next_states).max(1)
            max_next_q_values = (
                self.target_network(next_states)
                .gather(1, max_next_action.unsqueeze(1))
                .squeeze()
            )
            target_q_values = (
                rewards + (1 - dones) * self.config.gamma * max_next_q_values
            )

        predicted_q_values = self.policy_network(states)
        predicted_q_values = predicted_q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze()
        loss = self.loss_fn(predicted_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def soft_update_target_network(self):
        for target_param, policy_param in zip(
            self.target_network.parameters(), self.policy_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
