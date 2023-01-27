# %%
# import gymnasium as gym
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.utils.util import ShellColor as sc
from src.utils.util import get_preprocessed_img

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils.util import print_env_info

#%%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("Breakout-v0")
print_env_info(env=env)

# %%
obs_space_shape = (1, 84, 84)
agent = DQNAgent(
    obs_space_shape=obs_space_shape,
    action_space_dims=env.action_space.n,
    is_atari=True,
)
agent.config.batch_size = 32
agent.config.n_episodes = 1000
print(agent.config)

# obs = env.reset()
# plt.title("Origin Image")
# plt.imshow(obs)
# print(f"Origin Image Shape: {obs.shape}")
# plt.show()

# pre_processed_obs = get_preprocessed_img(obs)
# print(pre_processed_obs.shape)
# plt.title("Preprocessed Image")
# plt.imshow(pre_processed_obs)
# print(f"Preprocessed Image Shape: {pre_processed_obs.shape}")
# plt.show()

# state = np.stack(
#     (pre_processed_obs, pre_processed_obs, pre_processed_obs, pre_processed_obs)
# )
# print(state.shape)
# plt.title("Preprocessed Image")
# plt.imshow(state[0])
# print(f"Preprocessed Image Shape: {state[0][:].shape}")
# plt.show()

# action = agent.get_action(state)
# action
# next_obs, reward, done, info = env.step(action)
# next_obs
# plt.title("Preprocessed Image")
# plt.imshow(next_obs)
# print(f"Preprocessed Image Shape: {next_obs.shape}")
# plt.show()

# %%
for episode in range(agent.config.n_episodes):
    obs = env.reset()
    # obs = np.transpose(np.array(obs), (2, 0, 1))
    # print(obs.shape)
    pre_processed_obs = get_preprocessed_img(obs)
    state = np.stack(
        (
            pre_processed_obs,
            # pre_processed_obs,
            # pre_processed_obs,
            # pre_processed_obs,
        )
    )

    done = False
    ep_ret = 0
    ep_len = 0
    while not done:
        # env.render()
        # print(state.shape)

        action = agent.get_action(state)

        next_obs, reward, done, info = env.step(action)

        next_pre_processed_obs = get_preprocessed_img(next_obs)
        next_state = np.stack(
            (
                next_pre_processed_obs,
                # next_pre_processed_obs,
                # next_pre_processed_obs,
                # next_pre_processed_obs,
            )
        )

        # plt.imshow(next_state[0])
        # print(f"Next Image Shape: {next_state.shape}")
        # plt.show()
        # print(action, reward)
        # # next_obs = np.transpose(np.array(next_obs), (2, 0, 1))
        # print(ep_ret)

        agent.store_transition(state, action, reward, next_state, done)

        if len(agent.memory.buffer) > 10:
            agent.update()

        state = next_state
        ep_ret += reward
        ep_len += 1

    agent.decay_epsilon()

    if (episode + 1) % agent.config.target_update:
        agent.q_target.load_state_dict(agent.q_predict.state_dict())

    if (episode == 0) or (((episode + 1) % 1) == 0):
        print(
            f"episode: {episode + 1} | ep_ret: {ep_ret:.4f} | epsilon: {agent.epsilon:.4f}"
        )

env.close()
