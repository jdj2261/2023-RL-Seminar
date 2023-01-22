import numpy as np
from src.commons.memory.base_memory import Memory
from src.utils.sum_tree import SumTree


class PrioritizedMemory(Memory):
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, memory_capacity) -> None:
        super().__init__(memory_capacity)
        self.memory = SumTree(capacity=memory_capacity)

    def store(self, state, action, reward, next_state, done):
        pass

    def sample(self, batch_size):
        pass

    def update_priority(self):
        pass

    def clear(self):
        self.memory.clear()

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.e) ** self.a
