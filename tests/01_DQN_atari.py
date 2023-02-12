# %%
import gymnasium as gym

# import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils import util as rl_util

#%%
env = gym.make("PongNoFrameskip-v4")
env.seed(0)
rl_util.print_env_info(env=env)


# %%
config = rl_util.create_config()
config["n_episodes"] = 600
config["batch_size"] = 64
config["buffer_size"] = 100000
config["gamma"] = 0.99
config["target_update_frequency"] = 10000
config["lr"] = 0.00025

agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=True,
    config=config,
)
print(agent.config)
print(type(agent.memory))
#%%
############################ Plot ############################
# obs, _ = env.reset()
# plt.figure()
# plt.imshow(obs)
# plt.title("Original Frame")
# plt.show()
# #%%
# plt.figure()
# plt.imshow(rl_util.get_preprocessed_frame(obs, (30, -4, -12, 4), 84), cmap="gray")
# plt.title("Pre Processed image")
# plt.show()
# #%%
# plt.plot([agent.decay_epsilon(i) for i in range(1000)])
#%%
# env = gym.make("PongNoFrameskip-v4")
# obs, _ = env.reset()
# state = rl_util.stack_frames(None, obs, True)
# print(state.shape)
# for j in range(200):
#     # env.render()
#     action = agent.select_action(state)
#     next_state, reward, done, _, _ = env.step(action)
#     state = rl_util.stack_frames(state, next_state, False)
#     if done:
#         break
# env.close()
#%%
save_dir = "result/DQN/atari/"
rl_util.create_directory(save_dir)
save_model_name = ""
# %%
rewards_window = deque([], maxlen=20)
rewards = []
losses = []
is_start = True

for i_episode in tqdm(range(1, agent.config.n_episodes)):
    state, _ = env.reset()
    state = rl_util.stack_frames(None, state, True)
    score = 0
    eps = agent.decay_epsilon(i_episode)
    t_step = 0
    avg_loss = 0

    while True:
        # env.render()
        action = agent.select_action(state, eps)
        next_state, reward, done, _, info = env.step(action)
        score += reward
        next_state = rl_util.stack_frames(state, next_state, False)
        agent.store_transition(state, action, reward, next_state, done)

        t_step = t_step % agent.config.target_update_frequency

        if t_step == 0:
            if len(agent.memory.replay_buffer) > 20000:
                if is_start:
                    print("Start learning...")
                    is_start = False
                loss = agent.update()
                avg_loss += loss

        state = next_state
        t_step += 1
        if done:
            break

    rewards.append(score)
    rewards_window.append(score)

    avg_loss = avg_loss / t_step
    losses.append(avg_loss)
    mean_rewards = np.mean(rewards_window)
    if i_episode % 10 == 0:
        print(
            f"episode: {i_episode} | average score: {mean_rewards:.4f} | loss: {avg_loss:.4f} | epsilon: {eps:.4f}"
        )

    # if i_episode % 100 == 0:
    #     print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, mean_rewards))
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     plt.plot(np.arange(len(rewards)), rewards)
    #     plt.ylabel("Score")
    #     plt.xlabel("Episode #")
    #     plt.show()

env.close()

# %%
# fig, ax = rl_util.init_2d_figure("Reward")
# rl_util.plot_graph(
#     ax,
#     rewards,
#     title="reward",
#     ylabel="reward",
#     save_dir_name=save_dir,
#     is_save=True,
# )
# rl_util.show_figure()
# fig, ax = rl_util.init_2d_figure("Loss")
# rl_util.plot_graph(
#     ax, losses, title="loss", ylabel="loss", save_dir_name=save_dir, is_save=True
# )
# rl_util.show_figure()
# %%

# env = gym.make("ALE/Pong-v5", render_mode="human")
# env = GrayScaleObservation(env, keep_dim=True)
# env = ResizeObservation(env, 84)
# env = FrameStack(env, 4)
# for ep in range(10):
#     obs, _ = env.reset()
#     obs = np.squeeze(obs, axis=-1)
#     total_reward = 0
#     while True:
#         env.render()
#         state = torch.tensor(
#             obs, dtype=torch.float, device=agent.config.device
#         ).unsqueeze(0)
#         action = torch.argmax(agent.policy_network(state)).item()
#         next_obs, reward, terminated, info = env.step(action)
#         next_obs = np.squeeze(next_obs, axis=-1)
#         done = terminated
#         total_reward += reward
#         obs = next_obs
#         if done:
#             break
#     print(total_reward)
# env.close()
