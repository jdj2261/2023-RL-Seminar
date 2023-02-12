from collections import deque, namedtuple
from abc import ABCMeta, abstractmethod


class Memory(metaclass=ABCMeta):
    def __init__(self, buffer_size: int) -> None:
        self.replay_buffer = []
        self.buffer_size = buffer_size

    @abstractmethod
    def store(self, state, action, reward, next_state, done):
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int):
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        for i in self._storage:
            yield i

    def __repr__(self) -> str:
        return f"memory size : {len(self._storage)}"
