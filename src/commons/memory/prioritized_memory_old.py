import numpy as np
import random
from src.commons.memory.base_memory import Memory
from src.utils.sum_tree import SumTree


class PrioritizedMemory(Memory):
    def __init__(self, buffer_size, alpha=0.6, beta=0.4) -> None:
        super().__init__(buffer_size)
        self.replay_buffer = SumTree(capacity=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.e = 0.01
        self.beta_increment_per_sampling = 0.001
        self.current_length = 0

    def store(self, state, action, reward, next_state, done):
        if self.current_length == 0:
            priority = 1.0
        else:
            priority = self.replay_buffer.tree.max()
        self.current_length = self.current_length + 1
        experience = (state, action, np.array([reward]), next_state, done)
        self.replay_buffer.add(priority, experience)

    def sample(self, batch_size):
        batch_idx, batch, IS_weights = [], [], []
        segment = self.replay_buffer.total() / batch_size
        p_sum = self.replay_buffer.tree[0]

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.replay_buffer.get(s)

            batch_idx.append(idx)
            batch.append(data)
            prob = p / p_sum
            IS_weight = (self.replay_buffer.total() * prob) ** (-self.beta)
            IS_weights.append(IS_weight)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (
            np.array(state_batch),
            np.array(action_batch),
            np.array(reward_batch),
            np.array(next_state_batch),
            np.array(done_batch),
            batch_idx,
            IS_weights,
        )

    def update_priority(self, idx, td_error):
        priority = td_error**self.alpha + 1e-6
        self.replay_buffer.update(idx, priority)

    def clear(self):
        pass

    def __len__(self):
        return self.current_length
