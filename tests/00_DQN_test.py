import gymnasium as gym
import torch
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils.util import print_env_info
from src.utils.plot import plot_graph


env = gym.make("CartPole-v1", render_mode="human")
test_agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
)
test_agent.q_predict.load_state_dict(torch.load("checkpoint.pth"))

for i in range(1000):
    obs, info = env.reset()
    for j in range(100):
        action = test_agent.get_action(obs)
        env.render()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            break

env.close()
