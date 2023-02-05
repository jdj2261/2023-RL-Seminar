import numpy as np
import random
from src.commons.memory.base_memory import Memory
from src.utils.sum_tree import SumTree


class PrioritizedMemory(Memory):
    e = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, memory_capacity) -> None:
        super().__init__(memory_capacity)
        self.replay_buffer = SumTree(capacity=memory_capacity)
        self.current_length = 0

    def store(self, state, action, reward, next_state, done):
        priority = 1.0 if self.current_length is 0 else self.replay_buffer.tree.max()
        self.current_length = self.current_length + 1
        experience = (state, action, np.array([reward]), next_state, done)
        self.replay_buffer.add(priority, experience)

    def sample(self, batch_size):
        batch_idx, batch = [], []
        priorities = []
        segment = self.replay_buffer.total() / batch_size
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            while True:
                s = random.uniform(a, b)
                (idx, p, transition) = self.replay_buffer.get(s)
                if not isinstance(transition, int):
                    break

            priorities.append(p)
            batch.append(transition)
            batch_idx.append(idx)

        sampling_probabilities = priorities / self.replay_buffer.total()
        is_weight = np.power(
            self.replay_buffer.n_entries * sampling_probabilities, -self.beta
        )
        is_weight /= is_weight.max()

        return (
            batch,
            batch_idx,
            is_weight,
        )

    def update_priority(self, idx, td_error):
        priority = td_error**self.alpha + 1e-6
        self.replay_buffer.update(idx, priority)

    def clear(self):
        self.replay_buffer = None

    def __len__(self):
        return self.current_length
