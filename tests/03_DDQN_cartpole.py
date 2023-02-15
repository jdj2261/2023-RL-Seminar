# %%
import gymnasium as gym
import torch
import numpy as np
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.ddqn_agent import DDQNAgent
from src.utils import util as rl_util

# %%
env = gym.make("CartPole-v1")
rl_util.print_env_info(env=env)
env.observation_space.shape
# %%
agent = DDQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
)
# %%
agent.config.n_episodes = 1000
agent.config.target_update_frequency = 100
agent.config.buffer_size = 10000
agent.config.batch_size = 64
print(agent.config)
print(type(agent.memory))
# %%
