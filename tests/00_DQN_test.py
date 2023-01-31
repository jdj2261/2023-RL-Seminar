import gymnasium as gym
import torch
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils import util as rl_util


# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
rl_util.print_env_info(env=env)
test_agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
)
print(test_agent.config)

save_dir = "result/DQN/cartpole/"
file_name = "checkpoint_2023_02_01_02_26_24.pt"
test_agent.q_predict.load_state_dict(torch.load(save_dir + file_name))
for episode in range(1000):
    obs, info = env.reset()
    done = False
    ep_ret = 0
    ep_len = 0

    while not done:
        action = test_agent.get_action(obs)
        # env.render()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        test_agent.store_transition(obs, action, reward, next_obs, done)

        torch.cuda.empty_cache()
        ep_ret += reward
        ep_len += 1
        obs = next_obs
    test_agent.decay_epsilon()
    if (episode == 0) or (((episode + 1) % 1) == 0):
        print(f"episode: {episode + 1} | ep_ret: {ep_ret:.4f}")
env.close()
