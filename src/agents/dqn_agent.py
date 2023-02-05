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
        is_fixed_seed: bool = False,
        is_atari: bool = True,
        memory_type="uniform",
        config: dict = {},
    ) -> None:
        super().__init__(obs_space_shape, action_space_dims, config)

        if is_fixed_seed:
            np.random.seed(self.config.seed)

        self.q_target, self.q_predict = self._get_q_models(
            obs_space_shape, action_space_dims, self.config.device, is_atari
        )

        self.use_priority = False
        if "priority" in memory_type:
            self.use_priority = True

        # Memory Type
        self._memory = self._get_memory()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_predict.parameters(), lr=self.config.lr, amsgrad=True
        )

        # Loss function
        self.loss = None
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss()

    def _get_q_models(self, obs_space_shape, action_space_dims, device, is_atari):
        if not is_atari:
            q_target = Model(obs_space_shape, action_space_dims).to(device)
            q_predict = Model(obs_space_shape, action_space_dims).to(device)
        else:
            q_target = CNNModel(obs_space_shape, action_space_dims).to(
                device, non_blocking=True
            )
            q_predict = CNNModel(obs_space_shape, action_space_dims).to(
                device, non_blocking=True
            )
        q_target.load_state_dict(q_predict.state_dict())

        return q_target, q_predict

    def _get_memory(self):
        memory = ReplayMemory(self.config.memory_capacity)
        if self.use_priority:
            memory = PrioritizedMemory(self.config.memory_capacity)
        return memory

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(np.arange(self.action_space_dims), 1)[0]
        else:
            with torch.no_grad():
                state = torch.tensor(
                    state, dtype=torch.float, device=self.config.device
                )
                q_values = self.q_predict(state)
                action = torch.argmax(q_values).item()
        return action

    def store_transition(self, state, action, reward, next_state, done) -> None:
        transition = (state, action, reward, next_state, done)
        self._memory.store(transition)

    def update(self):
        if not self.use_priority:
            mini_batch = self._memory.sample(self.config.batch_size)
            states, actions, rewards, next_states, dones = self._get_tensor_batch(
                mini_batch
            )
            cur_Q = self.q_predict(states)
            next_Q = self.q_target(next_states)
            td_target = (
                rewards
                + self.config.gamma * torch.max(next_Q, dim=1)[0].reshape(-1, 1) * dones
            )
            loss = self.loss_fn(td_target.detach(), cur_Q.gather(1, actions))

        else:
            batch, batch_indexes, is_weights = self._memory.sample(
                self.config.batch_size
            )
            batch = np.array(batch).transpose()
            states, actions, rewards, next_states, dones = batch
            # experiences = [self._memory.experience(*n) for n in batch]
            state_batch = torch.FloatTensor(np.array(states)).to(self.config.device)
            action_batch = torch.LongTensor(actions).to(self.config.device)
            reward_batch = torch.FloatTensor(np.array(rewards)).to(self.config.device)
            next_state_batch = torch.FloatTensor(np.array(next_states)).to(
                self.config.device
            )
            done_batch = torch.FloatTensor(dones).to(self.config.device)
            is_weights = torch.FloatTensor(is_weights).to(self.config.device)

            current_q_values = self.q_predict(state_batch).gather(1, action_batch)
            next_q_values = self.q_target(next_state_batch).max(dim=1).values.detach()
            expected_Q = reward_batch.squeeze(1) + self.config.gamma * next_q_values
            td_errors = torch.pow(current_q_values - expected_Q, 2) * is_weights
            loss = (is_weights * F.mse_loss(expected_Q, current_q_values)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_target_update(self.q_predict, self.q_target)

        # if self.use_priority:
        #     for idx, td_error in zip(batch_indexes, td_errors.cpu().detach().numpy()):
        #         self._memory.update_priority(idx, td_error)

        self.loss = loss.item()

    def soft_target_update(self, model, model_p):
        for param_target, param in zip(model_p.parameters(), model.parameters()):
            param_target.data.copy_(
                param_target.data * (1.0 - 0.01) + param.data * 0.01
            )

    def _get_tensor_batch(self, mini_batch):
        s_list, r_list, a_list, s_p_list, done_list = [], [], [], [], []
        for sample in mini_batch:
            s_list.append(sample[0])
            s_p_list.append(sample[3])

            r_list.append([sample[2]])
            a_list.append([sample[1]])
            done_list.append([0]) if sample[-1] else done_list.append([1])
        states = np.vstack(s_list)
        next_states = np.vstack(s_p_list)
        states = torch.tensor(states, dtype=torch.float32, device=self.config.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.config.device
        )
        actions = torch.tensor(a_list, device=self.config.device).reshape(-1, 1)
        rewards = torch.tensor(
            r_list, dtype=torch.float, device=self.config.device
        ).reshape(-1, 1)
        dones = torch.tensor(
            done_list, dtype=torch.float, device=self.config.device
        ).reshape(-1, 1)

        # states, actions, rewards, next_states, dones = (
        #     mini_batch[0],
        #     mini_batch[1],
        #     mini_batch[2],
        #     mini_batch[3],
        #     mini_batch[4],
        # )

        # states = np.vstack(states)
        # states = torch.tensor(states, dtype=torch.float32, device=self.config.device)

        # actions = torch.tensor(
        #     list(actions), dtype=torch.long, device=self.config.device
        # ).view(-1, 1)
        # rewards = torch.tensor(list(rewards), device=self.config.device)
        # next_states = np.vstack(next_states)
        # next_states = torch.tensor(
        #     next_states, dtype=torch.float32, device=self.config.device
        # )

        # dones = torch.tensor(list(dones), device=self.config.device)

        return states, actions, rewards, next_states, dones

    @property
    def memory(self):
        return self._memory
