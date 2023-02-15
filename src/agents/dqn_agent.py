import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from src.agents.base_agent import Agent
from src.commons.memory import PrioritizedMemory, ReplayMemory
from src.commons.model import CNNModel, Model


class DQNAgent(Agent):
    def __init__(
        self,
        obs_space_shape: tuple,
        action_space_dims: int,
        is_fixed_seed: bool = True,
        is_atari: bool = True,
        use_double_dqn=False,
        config: dict = {},
    ) -> None:
        super().__init__(obs_space_shape, action_space_dims, config)

        if is_fixed_seed:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

        self.target_network, self.policy_network = self._get_q_models(
            obs_space_shape, action_space_dims, self.config.device, is_atari
        )
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.lr)
        if use_double_dqn:
            print("Double DQN Agent")
        else:
            print("DQN Agent")

        self.use_double_dqn = use_double_dqn

        self.use_priority = False
        if "priority" in self.config.buffer_type:
            print(self.config.buffer_type)
            self.use_priority = True

        self.is_atari = is_atari

        if self.is_atari:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        # Memory Type
        self.memory = self._get_memory()
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

    def _get_memory(self):
        memory = ReplayMemory(self.config.buffer_size)
        if self.use_priority:
            memory = PrioritizedMemory(self.config.buffer_size)
        return memory

    def select_action(self, state: np.ndarray, eps=0.0):
        assert (
            isinstance(state) is np.ndarray
        ), "The type of state must be np.ndarray!!!"

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
        if not self.use_priority:
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.config.batch_size
            )
        else:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                idxs,
                IS_weights,
            ) = self.memory.sample(self.config.batch_size)

        if self.use_priority:
            states = np.array(states)
            next_states = np.array(next_states)

            states = torch.from_numpy(states).float().to(self.config.device)
            actions = torch.from_numpy(actions).long().to(self.config.device)
            rewards = torch.from_numpy(rewards).float().to(self.config.device)
            next_states = torch.from_numpy(next_states).float().to(self.config.device)
            dones = torch.from_numpy(dones).float().to(self.config.device)
            IS_weights = torch.FloatTensor(IS_weights).to(self.config.device)

            next_q_values = self.target_network(next_states)
            max_next_q_values, _ = next_q_values.max(1)
            target_q_values = (
                rewards + (1 - dones) * self.config.gamma * max_next_q_values
            )

            input_q_values = self.policy_network(states)
            input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

            loss = (input_q_values - target_q_values).pow(2)
            prios = loss + 1e-5
            loss = (loss * IS_weights.unsqueeze(1)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.memory.update_priorities(idxs, prios.data.cpu().numpy())
            del states
            del next_states
            return loss
        else:
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
                if self.use_double_dqn:
                    _, max_next_action = self.policy_network(next_states).max(1)
                    max_next_q_values = (
                        self.target_network(next_states)
                        .gather(1, max_next_action.unsqueeze(1))
                        .squeeze()
                    )
                else:
                    next_q_values = self.target_network(next_states)
                    max_next_q_values, _ = next_q_values.max(1)
                target_q_values = (
                    rewards + (1 - dones) * self.config.gamma * max_next_q_values
                )

            input_q_values = self.policy_network(states)
            input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()
            loss = self.loss_fn(input_q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            del states
            del next_states
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

    # @property
    # def memory(self):
    #     return self._memory
