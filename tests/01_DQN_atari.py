# %%
import gymnasium as gym

# import gym
import torch
import numpy as np

from src.utils.util import ShellColor as sc
from src.utils.util import get_preprocessed_img

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils import util as rl_util

#%%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("ALE/Pong-v5")
rl_util.print_env_info(env=env)

# %%
obs_space_shape = (4, 84, 84)
agent = DQNAgent(
    obs_space_shape=obs_space_shape,
    action_space_dims=env.action_space.n,
    is_atari=True,
)
agent.config.batch_size = 64
agent.config.n_episodes = 5000
agent.config.memory_capacity = 10000
print(agent.config)

# #%%
# obs = env.reset()
# plt.title("Origin Image")
# plt.imshow(obs[0])
# print(f"Origin Image Shape: {obs[0].shape}")
# plt.show()

# #%%
# print(obs[0].shape)
# pre_processed_obs = get_preprocessed_img(obs[0], 84, 84)
# print(pre_processed_obs.shape)
# plt.title("Preprocessed Image")
# plt.imshow(pre_processed_obs)
# print(f"Preprocessed Image Shape: {pre_processed_obs.shape}")
# plt.show()

# #%%
# state = np.stack(
#     (pre_processed_obs, pre_processed_obs, pre_processed_obs, pre_processed_obs)
# )
# print(state.shape)
# plt.title("Preprocessed Image")
# plt.imshow(state[0])
# print(f"Preprocessed Image Shape: {state[0][:].shape}")
# plt.show()

# #%%
# action = agent.get_action(state)
# print(action)
# next_obs, reward, terminated, truncated, info = env.step(action)
# print(next_obs.shape)
# plt.title("Preprocessed Image")
# plt.imshow(next_obs)
# print(f"Preprocessed Image Shape: {next_obs.shape}")
# plt.show()

# %%
rewards = []
losses = []
for episode in range(agent.config.n_episodes):
    obs = env.reset()

    pre_processed_obs = get_preprocessed_img(
        obs[0], obs_space_shape[1], obs_space_shape[2]
    )
    state = np.stack(
        (
            pre_processed_obs,
            pre_processed_obs,
            pre_processed_obs,
            pre_processed_obs,
        )
    )

    done = False
    ep_ret = 0
    ep_len = 0
    frame = 0
    avg_loss = 0
    while not done:
        # env.render()
        # print(state.shape)

        action = agent.get_action(state)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # reward = 1 if reward > 0 else -0.1
        next_pre_processed_obs = get_preprocessed_img(
            next_obs, obs_space_shape[1], obs_space_shape[2]
        )
        next_state = np.stack(
            (
                next_pre_processed_obs,
                next_pre_processed_obs,
                next_pre_processed_obs,
                next_pre_processed_obs,
            )
        )

        # plt.imshow(next_state[0])
        # print(f"Next Image Shape: {next_state.shape}")
        # plt.show()
        # print(action, reward)
        # # next_obs = np.transpose(np.array(next_obs), (2, 0, 1))
        # print(ep_ret)

        agent.store_transition(state, action, reward, next_state, done)

        if len(agent.memory.buffer) > 2000:
            agent.update()
            avg_loss += agent.loss

        state = next_state
        ep_ret += reward
        ep_len += 1
        frame += 1

    agent.decay_epsilon()
    avg_loss /= frame

    if (episode + 1) % agent.config.target_update:
        agent.q_target.load_state_dict(agent.q_predict.state_dict())

    if (episode == 0) or (((episode + 1) % 1) == 0):
        print(
            f"episode: {episode + 1} | ep_ret: {ep_ret:.4f} | loss: {avg_loss:.4f} | epsilon: {agent.epsilon:.4f}"
        )
    rewards.append(ep_ret)
    losses.append(avg_loss)
env.close()

# %%
save_dir = "result/DQN/atari/"
rl_util.create_directory(save_dir)

fig, ax = rl_util.init_2d_figure("test")
rl_util.plot_graph(
    ax, rewards, title="reward", ylabel="reward", save_dir_name=save_dir, is_save=True
)
rl_util.plot_graph(
    ax, losses, title="loss", ylabel="loss", save_dir_name=save_dir, is_save=True
)
rl_util.show_figure()
current_time = rl_util.get_current_time_string()
save_model_name = save_dir + "checkpoint_" + current_time + ".pt"
torch.save(agent.q_predict.state_dict(), save_model_name)

# %%
