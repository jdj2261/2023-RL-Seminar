import numpy as np
from src.commons.memory.base_memory import Memory


class ReplayMemory(Memory):
    def __init__(self, buffer_size=10000) -> None:
        super().__init__(buffer_size)

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.randint(0, len(self.replay_buffer), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            transition = self.replay_buffer[i]
            state, action, reward, next_state, done = transition
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            next_states.append(np.array(next_state, copy=False))
            dones.append(np.array(done, copy=False))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
