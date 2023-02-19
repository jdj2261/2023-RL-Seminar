import numpy as np
import random
from src.commons.memory.base_memory import Memory
from src.utils.sum_tree import SumTree


class PrioritizedMemory(Memory):
    def __init__(self, buffer_size: int) -> None:
        super().__init__(buffer_size)
        self.replay_buffer = SumTree(capacity=buffer_size)
        self.alpha = 0.6
        self.beta = 0.4
        self.max_priority = 1.0

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            p=self.max_priority, data=(state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idxs = []
        priorities = []
        segment = self.replay_buffer.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.replay_buffer.get(s)
            idxs.append(idx)
            priorities.append(p)

            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_states.append(np.array(next_state, copy=False))
            dones.append(np.array(done, copy=False))

        sampling_probability = priorities / self.replay_buffer.total()
        importance_sampling_weights = np.power(
            self.replay_buffer.n_entries * sampling_probability, -self.beta
        )
        importance_sampling_weights /= importance_sampling_weights.max()
        batch = (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
        return batch, idxs, importance_sampling_weights

    def update_priority(self, idx, td_error):
        priority = td_error**self.alpha + 1e-6
        self.replay_buffer.update(idx, priority)
