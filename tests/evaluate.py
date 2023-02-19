import gymnasium as gym
import torch
import argparse
import os

current_dir_path = os.path.abspath(os.path.dirname(__file__))
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents import *

# python evaluate.py -e CartPole-v1 -a DQN -c result/DQN/cartpole/CartPole-v1_2023_02_19_04_43_28.pt
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-e",
        "--environment",
        type=str,
        default="PongNoFrameskip-v4",
        choices=["PongNoFrameskip-v4", "CartPole-v1"],
        help="Choose Environment [PongNoFrameskip-v4, CartPole-v1]",
    )
    ap.add_argument(
        "-a",
        "--agent",
        type=str,
        default="DQN",
        choices=["DQN", "DQNPER", "DDQN", "DDQNPER"],
        help="Choose Agent [DQN, DQNPER, DDQN, DDQNPER]",
    )
    ap.add_argument(
        "-c", "--checkpoint", type=str, required=True, help="Checkpoint for agent"
    )

    opt = ap.parse_args()
    return opt


opt = get_args()

env = gym.make(opt.environment, render_mode="human")

is_atari = False
if "cartpole" not in str(opt.environment).lower():
    env = gym.wrappers.AtariPreprocessing(
        env=env, terminal_on_life_loss=True, grayscale_obs=True, noop_max=0
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    is_atari = True

if str(opt.agent).upper() == "DQN":
    test_agent = DQNAgent(
        obs_space_shape=env.observation_space.shape,
        action_space_dims=env.action_space.n,
        is_atari=is_atari,
    )
if str(opt.agent).upper() == "DQNPER":
    test_agent = DQNPerAgent(
        obs_space_shape=env.observation_space.shape,
        action_space_dims=env.action_space.n,
        is_atari=is_atari,
    )
if str(opt.agent).upper() == "DDQN":
    test_agent = DDQNAgent(
        obs_space_shape=env.observation_space.shape,
        action_space_dims=env.action_space.n,
        is_atari=is_atari,
    )
if str(opt.agent).upper() == "DDQNPER":
    test_agent = DDQNPerAgent(
        obs_space_shape=env.observation_space.shape,
        action_space_dims=env.action_space.n,
        is_atari=is_atari,
    )

save_model_name = opt.checkpoint
test_agent.policy_network.load_state_dict(
    torch.load(os.path.join(current_dir_path, save_model_name))
)

for i_episode in range(1):
    state, _ = env.reset()
    test_reward = 0
    while True:
        env.render()
        action = test_agent.select_action(state, 0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        test_reward += reward
        state = next_state
        done = terminated or truncated
        if done:
            break
    print(f"{i_episode+1} episode Total Reward: {test_reward}")
env.close()
