from collections import deque
from abc import ABCMeta, abstractmethod


class Memory(metaclass=ABCMeta):
    def __init__(self, buffer_size: int) -> None:
        self.replay_buffer = deque([], maxlen=buffer_size)
        self.buffer_size = buffer_size

    @abstractmethod
    def store(self, state, action, reward, next_state, done):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int):
        raise NotImplementedError

    def __len__(self):
        return len(self.replay_buffer)

    def __iter__(self):
        for i in self.replay_buffer:
            yield i

    def __repr__(self) -> str:
        return f"memory size : {len(self.replay_buffer)}"
