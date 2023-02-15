import gymnasium as gym
import torch
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils import util as rl_util


env = gym.make("CartPole-v1")
config = rl_util.create_config()
config["batch_size"] = 32
config["buffer_size"] = 30000
config["gamma"] = 0.99
config["target_update_frequency"] = 4

test_agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
    config=config,
)
print(test_agent.config)

save_dir = "result/DQN/cartpole/"
file_name = "checkpoint_2023_02_13_09_22_54.pt"
print(save_dir + file_name)
test_agent.policy_network.load_state_dict(torch.load(save_dir + file_name))

# %%
# load the weights from file
env = gym.make("CartPole-v1", render_mode="human")
for ep in range(10):
    obs, _ = env.reset()
    total_reward = 0
    while True:
        env.render()
        state = torch.tensor(obs, dtype=torch.float, device=test_agent.config.device)
        action = torch.argmax(test_agent.policy_network(state)).item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        obs = next_obs
        if done:
            break
    print(total_reward)
env.close()
