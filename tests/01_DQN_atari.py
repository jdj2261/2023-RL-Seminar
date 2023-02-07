# %%
import gymnasium as gym

# import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import ResizeObservation

from src.utils.util import ShellColor as sc
from src.utils.util import get_preprocessed_img

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils import util as rl_util

#%%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("ALE/Pong-v5")
# env = gym.make("ALE/Pong-v5", render_mode="human")
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, 84)
env = FrameStack(env, 4)
rl_util.print_env_info(env=env)
# %%
stack_frame = env.observation_space.shape[0]

# %%
config = rl_util.create_config()
config["n_episodes"] = 3000
config["batch_size"] = 32
config["memory_capacity"] = 30000
config["gamma"] = 0.99
config["update_frequency"] = 4

agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=True,
    config=config,
)

print(agent.config)
print(type(agent.memory))
#%%

#%%
# obs, _ = env.reset()
# obs = np.squeeze(obs, axis=-1)
# # #%%
# print(obs)
# for i in range(stack_frame):
#     plt.title(f"{i+1} Origin Image")
#     plt.imshow(obs[i])
#     print(f"{i+1} Origin Image Shape: {obs[i].shape}")
#     plt.show()

# #%%
# action = agent.select_action(obs)
# next_obs, reward, terminated, truncated, info = env.step(action)
# for episode in range(5):
#     for i in range(stack_frame):
#         plt.title(f"{episode}episode | {i+1} Preprocessed Image")
#         plt.imshow(next_obs[i])
#         print(f"{episode}episode | {i+1} Preprocessed Image Shape: {next_obs[i].shape}")
#         plt.show()
#     print("=" * 60)
# %%
#%%
save_dir = "result/DQN/atari/"
rl_util.create_directory(save_dir)
save_model_name = ""
# %%
rewards = deque([], maxlen=5)
total_rewards = []
losses = []

for i_episode in range(agent.config.n_episodes):
    obs, info = env.reset()
    obs = np.squeeze(obs, axis=-1)
    avg_loss = 0
    len_game_progress = 0
    test_reward = 0
    while True:
        # env.render()
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.squeeze(next_obs, axis=-1)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)

        if len(agent.memory.replay_buffer) > 10000:
            agent.update()
            avg_loss += agent.loss

        obs = next_obs
        len_game_progress += 1
        test_reward += reward
        if done:
            break

    # print(test_reward)
    agent.decay_epsilon()
    avg_loss /= len_game_progress

    if (i_episode + 1) % 1 == 0:
        # test_reward = rl_util.atari_evaluate_agent(env, agent, num=1)
        print(
            f"episode: {i_episode} | cur_reward: {test_reward:.4f} | loss: {avg_loss:.4f} | epsilon: {agent.epsilon:.4f}"
        )
        rewards.append(test_reward)
        total_rewards.append(test_reward)
        losses.append(avg_loss)

    mean_rewards = np.mean(rewards)
    # if mean_rewards > 490:
    #     current_time = rl_util.get_current_time_string()
    #     save_model_name = save_dir + "checkpoint_" + current_time + ".pt"
    #     print(f"Save model {save_model_name} | episode is {(i_episode)}")
    #     torch.save(agent.q_predict.state_dict(), save_model_name)
    #     break

env.close()

# %%
fig, ax = rl_util.init_2d_figure("Reward")
rl_util.plot_graph(
    ax,
    total_rewards,
    title="reward",
    ylabel="reward",
    save_dir_name=save_dir,
    is_save=True,
)
rl_util.show_figure()
fig, ax = rl_util.init_2d_figure("Loss")
rl_util.plot_graph(
    ax, losses, title="loss", ylabel="loss", save_dir_name=save_dir, is_save=True
)
rl_util.show_figure()
# %%

env = gym.make("ALE/Pong-v5", render_mode="human")
env = GrayScaleObservation(env, keep_dim=True)
env = ResizeObservation(env, 84)
env = FrameStack(env, 4)
for ep in range(10):
    obs, _ = env.reset()
    obs = np.squeeze(obs, axis=-1)
    total_reward = 0
    while True:
        env.render()
        state = torch.tensor(
            obs, dtype=torch.float, device=agent.config.device
        ).unsqueeze(0)
        action = torch.argmax(agent.q_predict(state)).item()
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.squeeze(next_obs, axis=-1)
        done = terminated or truncated
        total_reward += reward
        obs = next_obs
        if done:
            break
    print(total_reward)
env.close()
