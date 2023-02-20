from abc import *
from src.utils.util import Config, get_device
from src.commons.model import CNNModel, Model


class Agent(metaclass=ABCMeta):
    def __init__(
        self,
        obs_space_shape: tuple,
        action_space_dims: int,
        config: dict,
    ) -> None:
        self.obs_space_shape = obs_space_shape
        self.action_space_dims = action_space_dims
        self.config = self._get_config(config)

        self.epsilon = self.config.epsilon_start
        self.epsilon_decay = 0.001

    @abstractmethod
    def select_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    def decay_epsilon(self, time_step):
        return max(
            self.config.epsilon_end,
            self.config.epsilon_start - time_step / self.config.epsilon_decay,
        )

    # def decay_epsilon(self):
    #     self.epsilon = max(self.config.epsilon_end, self.epsilon - self.epsilon_decay)

    @staticmethod
    def _get_q_models(obs_space_shape, action_space_dims, device, is_atari):
        if not is_atari:
            target_network = Model(obs_space_shape, action_space_dims).to(device)
            policy_network = Model(obs_space_shape, action_space_dims).to(device)
        else:
            target_network = CNNModel(obs_space_shape, action_space_dims).to(
                device, non_blocking=True
            )
            policy_network = CNNModel(obs_space_shape, action_space_dims).to(
                device, non_blocking=True
            )
        return target_network, policy_network

    @staticmethod
    def _get_config(config: dict) -> Config:
        init_config = Config()
        init_config.device = get_device()
        if config:
            init_config.n_episodes = config.get("n_episodes", init_config.n_episodes)
            init_config.max_steps = config.get("max_steps", init_config.max_steps)
            init_config.batch_size = config.get("batch_size", init_config.batch_size)
            init_config.gamma = config.get("gamma", init_config.gamma)
            init_config.lr = config.get("lr", init_config.lr)
            init_config.start_training_step = config.get(
                "start_training_step", init_config.start_training_step
            )
            init_config.learning_frequency = config.get(
                "learning_frequency", init_config.learning_frequency
            )
            init_config.epsilon_start = config.get("epsilon_start", init_config.epsilon_start)
            init_config.epsilon_end = config.get("epsilon_end", init_config.epsilon_end)
            init_config.epsilon_decay = config.get("epsilon_decay", init_config.epsilon_decay)
            init_config.seed = config.get("seed", init_config.seed)
            init_config.target_update_frequency = config.get(
                "target_update_frequency", init_config.target_update_frequency
            )
            init_config.buffer_size = config.get("buffer_size", init_config.buffer_size)
            init_config.mean_reward_bound = config.get(
                "mean_reward_bound", init_config.mean_reward_bound
            )
            init_config.print_frequency = config.get("print_frequency", init_config.print_frequency)
        return init_config
