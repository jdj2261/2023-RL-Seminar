from src.agents.base_agent import Agent


class DDQNAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, state):
        pass

    def store_transition(self, state, action, reward, next_state, done):
        pass

    def update(self):
        pass
