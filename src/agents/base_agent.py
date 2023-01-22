from abc import *
from src.utils.util import Config, get_device


class Agent(metaclass=ABCMeta):
    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        config: dict,
    ) -> None:
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims
        self.config = self._get_config(config)

        self.epsilon = self.config.epsilon_start

        # 0.001
        self.epsilon_decay = self.config.epsilon_start / (self.config.n_episodes)

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    # @abstractmethod
    # TODO (save / load file)
    def save_file(self, path: str):
        pass

    # @abstractmethod
    def load_file(self, path: str):
        pass

    def decay_epsilon(self):
        self.epsilon = max(self.config.epsilon_end, self.epsilon - self.epsilon_decay)

    @staticmethod
    def _get_config(config: dict) -> Config:
        init_config = Config()
        init_config.device = get_device()

        if not config:
            init_config.batch_size = config.get("batch_size", init_config.batch_size)
            init_config.gamma = config.get("gamma", init_config.gamma)
            init_config.lr = config.get("learning_rate", init_config.lr)
            init_config.seed = config.get("seed", init_config.seed)

            init_config.epsilon_start = config.get(
                "epsilon_start", init_config.epsilon_start
            )
            init_config.epsilon_end = config.get("epsilon_end", init_config.epsilon_end)

            init_config.memory_type = config.get("memory_type", init_config.memory_type)
            init_config.memory_capacity = config.get(
                "memory_capacity", init_config.memory_capacity
            )
            init_config.device = config.get("device", get_device())

        return init_config
