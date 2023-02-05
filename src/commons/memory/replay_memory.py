import numpy as np
import random
from src.commons.memory.base_memory import Memory


class ReplayMemory(Memory):
    def __init__(self, memory_capacity=10000) -> None:
        super().__init__(memory_capacity)

    def store(self, transition):
        self.replay_buffer.append(transition)

    def sample(self, batch_size: int) -> tuple:
        batch_size = (
            batch_size
            if len(self.replay_buffer) > batch_size
            else len(self.replay_buffer)
        )
        # indices = np.random.choice(len(self.replay_buffer), size=batch_size)
        # batch_samples = [self.replay_buffer[index] for index in indices]
        mini_batch = random.sample(self.replay_buffer, batch_size)
        return mini_batch

    def clear(self):
        self.replay_buffer.clear()
