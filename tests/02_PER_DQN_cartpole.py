# %%
import gymnasium as gym
import torch
import numpy as np
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils import util as rl_util

#%%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
rl_util.print_env_info(env=env)
env.observation_space.shape

# %%
agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
    buffer_type="priority",
)
agent.config.n_episodes = 1000
agent.config.target_update_frequency = 100
agent.config.buffer_size = 10000
agent.config.batch_size = 64
print(agent.config)
print(type(agent._memory))

#%%
save_dir = "result/PER/cartpole/"
rl_util.create_directory(save_dir)
save_model_name = ""

# %%
rewards = []
losses = []
max_steps = 500
for i_episode in range(agent.config.n_episodes):
    obs, info = env.reset()
    done = False
    ep_ret = 0
    ep_len = 0
    avg_loss = 0
    print(i_episode)
    for step in range(max_steps):
        # env.render()
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        # print(agent.memory)
        ep_ret += reward
        if len(agent.memory) > 32:
            agent.update()
            avg_loss += agent.loss

        if done or step == max_steps - 1:
            losses.append(ep_ret)
            # print("Episode " + str(episode) + ": " + str(episode_reward))
            break

        obs = next_obs
        ep_len += 1

    agent.decay_epsilon()
    avg_loss /= ep_len

    if (i_episode + 1) % agent.config.target_update_frequency:
        agent.q_target.load_state_dict(agent.policy_network.state_dict())

    if (i_episode == 0) or (((i_episode + 1) % 1) == 0):
        print(
            f"episode: {i_episode + 1} | ep_ret: {ep_ret:.4f} | loss: {avg_loss:.4f} | epsilon: {agent.epsilon:.4f}"
        )
        rewards.append(ep_ret)
        losses.append(avg_loss)

    if np.mean(rewards[-min(10, len(rewards)) :]) > 490 and i_episode % 100 == 0:
        current_time = rl_util.get_current_time_string()
        save_model_name = save_dir + "checkpoint_" + current_time + ".pt"
        print(f"Save model {save_model_name} | episode is {i_episode}")
        torch.save(agent.policy_network.state_dict(), save_model_name)

env.close()

#%%
fig, ax = rl_util.init_2d_figure("Reward")
rl_util.plot_graph(
    ax, rewards, title="reward", ylabel="reward", save_dir_name=save_dir, is_save=True
)
rl_util.show_figure()
fig, ax = rl_util.init_2d_figure("Loss")
rl_util.plot_graph(
    ax, losses, title="loss", ylabel="loss", save_dir_name=save_dir, is_save=True
)
rl_util.show_figure()

# %%
# load the weights from file
env = gym.make("CartPole-v1", render_mode="human")
test_agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
)
if not save_model_name:
    current_time = rl_util.get_current_time_string()
    save_model_name = save_dir + "checkpoint_" + current_time + ".pt"
    print(f"Save model {save_model_name} | episode is {i_episode}")
    torch.save(agent.policy_network.state_dict(), save_model_name)

    print(f"load {save_model_name} model ")
    test_agent.policy_network.load_state_dict(torch.load(save_model_name))

    for episode in range(1000):
        obs, info = env.reset()
        done = False
        ep_ret = 0

        while not done:
            action = test_agent.select_action(obs)
            env.render()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        if (episode == 0) or (((episode + 1) % 1) == 0):
            print(f"episode: {episode + 1} | ep_ret: {ep_ret:.4f}")

    env.close()

# %%
