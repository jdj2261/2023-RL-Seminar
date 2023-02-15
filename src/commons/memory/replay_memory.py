import numpy as np
from src.commons.memory.base_memory import Memory


# class ReplayMemory(Memory):
#     def __init__(self, buffer_size=10000) -> None:
#         super().__init__(buffer_size)

#     def store(self, state, action, reward, next_state, done):
#         self.replay_buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size: int) -> tuple:
#         batch_size = (
#             batch_size
#             if len(self.replay_buffer) > batch_size
#             else len(self.replay_buffer)
#         )

#         indices = np.random.randint(0, len(self.replay_buffer) - 1, size=batch_size)
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
#         )

#     def clear(self):
#         self.replay_buffer.clear()


class ReplayMemory(Memory):
    def __init__(self, buffer_size) -> None:
        super().__init__(buffer_size)
        self._next_idx = 0

    def __len__(self):
        return len(self.replay_buffer)

    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if self._next_idx >= len(self.replay_buffer):
            self.replay_buffer.append(transition)
        else:
            self.replay_buffer[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self.buffer_size

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            transition = self.replay_buffer[i]
            state, action, reward, next_state, done = transition
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self.replay_buffer) - 1, size=batch_size)
        return self._encode_sample(indices)
