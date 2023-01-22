from collections import deque, namedtuple
from abc import ABCMeta, abstractmethod


class Memory(metaclass=ABCMeta):
    def __init__(self, memory_capacity: int) -> None:
        self.buffer = deque(maxlen=memory_capacity)
        self.memory_capacity = memory_capacity
        self.experience = namedtuple(
            "Experience",
            ("state", "action", "reward", "next_state", "done"),
        )

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
        return len(self.buffer)

    def __iter__(self):
        for i in self.buffer:
            yield i

    def __repr__(self) -> str:
        return f"memory size : {len(self.buffer)}"
