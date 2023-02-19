import numpy as np
import random
from src.commons.memory.base_memory import Memory
from src.utils.sum_tree import SumTree


class PrioritizedMemory(Memory):
    def __init__(self, buffer_size: int) -> None:
        super().__init__(buffer_size)
        self.replay_buffer = SumTree(capacity=buffer_size)
        eps = 1e-2
        self.beta = 0.4
        self.max_priority = eps

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            p=self.max_priority, data=(state, action, reward, next_state, done)
        )

    def sample(self, batch_size: int):
        batch = []
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
            batch.append(data)

        sampling_probability = priorities / self.replay_buffer.total()
        importance_sampling_weights = np.power(
            self.replay_buffer.n_entries * sampling_probability, -self.beta
        )
        importance_sampling_weights /= importance_sampling_weights.max()

        return batch, idxs, importance_sampling_weights

    def update(self):
        pass


# class PrioritizedMemory(Memory):
#     def __init__(self, buffer_size) -> None:
#         super().__init__(buffer_size)
#         self.prob_alpha = 0.6
#         self.position = 0
#         self.priorities = np.zeros((buffer_size,), dtype=np.float32)

#     def store(self, state, action, reward, next_state, done):
#         transition = state, action, reward, next_state, done
#         max_prio = self.priorities.max() if self.replay_buffer else 1.0

#         if self.__len__() < self.buffer_size:
#             self.replay_buffer.append(transition)
#         if self.__len__() == self.buffer_size:
#             self.replay_buffer[self.position] = transition

#         self.priorities[self.position] = max_prio
#         self.position = (self.position + 1) % self.buffer_size

#     def sample(self, batch_size, beta=0.4):

#         if self.__len__() == self.buffer_size:
#             prios = self.priorities
#         else:
#             prios = self.priorities[: self.position]
#         probs = prios**self.prob_alpha
#         probs /= probs.sum()

#         indices = np.random.randint(0, len(self.replay_buffer) - 1, size=batch_size)

#         total = self.__len__()
#         weights = (total * probs[indices]) ** (-beta)
#         weights /= weights.max()
#         weights = np.array(weights, dtype=np.float32)

#         states, actions, rewards, next_states, dones = [], [], [], [], []
#         for i in indices:
#             data = self.replay_buffer[i]
#             state, action, reward, next_state, done = data
#             states.append(np.array(state, copy=False))
#             actions.append(action)
#             rewards.append(reward)
#             next_states.append(np.array(next_state, copy=False))
#             dones.append(done)

#         return (
#             np.array(states),
#             np.array(actions),
#             np.array(rewards),
#             np.array(next_states),
#             np.array(dones),
#             indices,
#             weights,
#         )

#     def update_priorities(self, batch_indices, batch_priorities):
#         for idx, prio in zip(batch_indices, batch_priorities):
#             self.priorities[idx] = prio

#     def clear(self):
#         pass

#     def __len__(self):
#         return len(self.replay_buffer)
