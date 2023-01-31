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
        is_fixed_seed: bool = True,
        is_atari: bool = True,
        memory_type="uniform",
        config: dict = {},
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
            self.q_target.load_state_dict(self.q_predict.state_dict())
        else:
            # TODO (atari or mujoco)
            self.q_target = CNNModel(obs_space_shape, action_space_dims).to(
                self.config.device, non_blocking=True
            )
            self.q_predict = CNNModel(obs_space_shape, action_space_dims).to(
                self.config.device, non_blocking=True
            )

        self.memory_type = memory_type

        self.use_priority = False
        if "priority" in memory_type:
            self.use_priority = True

        # Memory Type
        if self.use_priority:
            # prioritized
            self._memory = PrioritizedMemory(self.config.memory_capacity)
        else:
            # uniform
            self._memory = ReplayMemory(self.config.memory_capacity)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.q_predict.parameters(), lr=self.config.lr, amsgrad=True
        )

        # Loss function
        self.loss = None
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()

    def get_action(self, state):
        # epsilon greedy exploration
        if self.epsilon > np.random.random():
            action = np.random.choice(np.arange(self.action_space_dims), 1)[0]
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state, device=self.config.device, dtype=torch.float32
                ).unsqueeze(dim=0)
                # action = self.q_predict(state).max(1)[1].view(1, 1)
                q_values = self.q_predict(state)
                action = torch.argmax(q_values).item()
        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        transition = self._get_tensor_transition(
            state, action, reward, next_state, done
        )
        self._memory.store(*transition)

    def update(self):
        if len(self._memory.buffer) < self.config.batch_size:
            return

        # Sample random minibatch of transitions
        if self.use_priority:
            tree_idx, batch = self._memory.sample(self.batch_size)
        else:
            experiences = self._memory.sample(self.config.batch_size)
            batch = self._memory.experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])
        done_batch = torch.cat([s for s in batch.done if s is not None])

        next_q_values = self.q_target(next_state_batch).max(dim=1).values.detach()
        expected_q_values = reward_batch + self.config.gamma * next_q_values * (
            1 - done_batch.float()
        )
        current_q_values = self.q_predict(state_batch).gather(1, action_batch)
        loss = self.loss_fn(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for param in self.q_predict.parameters():
            param.grad.data.clamp_(-1, 1)

        # if self.use_priority:
        #     indices = np.arange(self.config.batch_size, dtype=np.int32)
        #     absolute_errors = np.abs(
        #         target_old[indices, action_batch] - target[indices, action_batch]
        #     )
        #     # Update priority
        #     self.MEMORY.batch_update(tree_idx, absolute_errors)

        self.loss = loss.item()

    def _get_tensor_transition(self, state, action, reward, next_state, done):
        state = torch.tensor(
            state, dtype=torch.float32, device=self.config.device
        ).unsqueeze(0)
        action = torch.tensor(
            [action], dtype=torch.long, device=self.config.device
        ).unsqueeze(0)
        reward = torch.tensor(reward, device=self.config.device).unsqueeze(0)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=self.config.device
        ).unsqueeze(0)
        done = torch.tensor(done, device=self.config.device).unsqueeze(0)

        return state, action, reward, next_state, done

    @property
    def memory_type(self):
        return self.config.memory_type

    @memory_type.setter
    def memory_type(self, memory_type):
        # Memory Type
        if "priority" in memory_type:
            # prioritized
            self._memory = PrioritizedMemory(self.config.memory_capacity)
        else:
            # uniform
            self._memory = ReplayMemory(self.config.memory_capacity)
        self.config.memory_type = memory_type

    @property
    def memory(self):
        return self._memory
