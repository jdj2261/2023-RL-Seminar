import numpy as np
import random
from src.commons.memory.base_memory import Memory


class ReplayMemory(Memory):
    def __init__(self, memory_capacity=10000) -> None:
        super().__init__(memory_capacity)

    def store(self, *transition):
        self.buffer.append(self.experience(*transition))

    def sample(self, batch_size: int) -> tuple:
        batch_size = batch_size if len(self.buffer) > batch_size else len(self.buffer)
        indices = np.random.choice(len(self.buffer), size=batch_size)
        batch_samples = [self.buffer[index] for index in indices]
        return batch_samples

    def clear(self):
        self.buffer.clear()
