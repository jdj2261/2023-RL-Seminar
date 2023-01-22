from src.agents.base_agent import Agent


# Example Class structure
class CustomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    # TODO
    def get_action(self, state):
        pass

    # TODO
    def store_transition(self, state, action, reward, next_state, done):
        pass

    # TODO
    def update(self):
        pass
