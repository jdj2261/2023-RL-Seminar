import numpy as np

from src.commons.memory.base_memory import Memory
from src.utils.sum_tree import SumTree


class PrioritizedMemory(Memory):
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    absolute_error_upper = 1.0

    def __init__(self, memory_capacity) -> None:
        super().__init__(memory_capacity)
        self.buffer = SumTree(capacity=memory_capacity)

    def store(self, *transition):
        max_priority = np.max(self.buffer.tree[-self.buffer.capacity :])

        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.buffer.add(max_priority, *transition)

    def sample(self, batch_size):
        pass

    def update_priority(self):
        pass

    def clear(self):
        self.buffer.clear()

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.e) ** self.a
