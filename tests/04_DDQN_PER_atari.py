#%%
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents import DDQNPerAgent
from src.utils import util as rl_util

#%%
env_name = "PongNoFrameskip-v4"
env = gym.make(env_name)
env = gym.wrappers.AtariPreprocessing(
    env=env,
    terminal_on_life_loss=True,
    grayscale_obs=True,
    scale_obs=False,
    noop_max=30,
)
env = gym.wrappers.FrameStack(env, num_stack=4)
rl_util.print_env_info(env=env)

#%%
config = rl_util.create_config()
config["print_frequency"] = 20
config["mean_reward_bound"] = 10

#%%
agent = DDQNPerAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=True,
    config=config,
)

print(agent.config)
print(type(agent.memory))

#%%
save_dir = "result/DDQN/per/atari/"
rl_util.create_directory(save_dir)
current_time = rl_util.get_current_time_string()
save_model_name = save_dir + env_name + "_" + current_time + ".pt"
#%%
obs, _ = env.reset()
episode_returns = []
episode_return = 0
episode_losses = []
episode_loss = 0
episode = 0
best_mean_return = -10000
is_start_train = True

start_time = time.time()
for t in range(agent.config.max_steps):
    epsilon = agent.decay_epsilon(t)
    action = agent.select_action(obs, epsilon)
    next_obs, reward, done, _, _ = env.step(action)
    agent.store_transition(obs, action, reward, next_obs, done)
    episode_return += reward

    if t > agent.config.start_training_step:
        if is_start_train:
            print(f"Start Training at timestep {t}...")
            is_start_train = False

        episode_loss += agent.update()
        if t % agent.config.target_update_frequency == 0:
            agent.update_target_network()

    if done:
        obs, _ = env.reset()
        episode += 1
        episode_returns.append(episode_return)
        episode_losses.append(episode_loss)
        mean_episode_return = np.mean(episode_returns[-agent.config.print_frequency :])
        mean_episode_loss = np.mean(episode_losses[-agent.config.print_frequency :])

        if episode % agent.config.print_frequency == 0:
            print(
                f"step: {t} | episode: {episode} | cur_return: {episode_return:.4f} | mean_return: {mean_episode_return:.4f} | best_mean_return: {best_mean_return:.4f} | loss: {episode_loss:.4f} | epsilon: {epsilon:.4f}"
            )

            if best_mean_return < mean_episode_return:
                torch.save(agent.policy_network.state_dict(), save_model_name)
                print(
                    f"Best mean return updated {best_mean_return:.3f} -> {mean_episode_return:.3f}, model saved"
                )
                best_mean_return = mean_episode_return
                if mean_episode_return > agent.config.mean_reward_bound:
                    print(f"Solved!")
                    break
        episode_return = 0
        episode_loss = 0
    else:
        obs = next_obs

end_time = time.time()
print(f"WorkingTime[{DDQNPerAgent.__name__}]: {end_time-start_time:.4f} sec\n")
#%%
fig, ax = rl_util.init_2d_figure("Reward")
rl_util.plot_graph(
    ax,
    episode_returns,
    title="reward",
    ylabel="reward",
    save_dir_name=save_dir,
    is_save=True,
)
rl_util.show_figure()
fig, ax = rl_util.init_2d_figure("Loss")
rl_util.plot_graph(
    ax,
    episode_losses,
    title="loss",
    ylabel="loss",
    save_dir_name=save_dir,
    is_save=True,
)
rl_util.show_figure()
#%%
env = gym.make(env_name, render_mode="human")
env = gym.wrappers.AtariPreprocessing(
    env=env, terminal_on_life_loss=True, grayscale_obs=True, noop_max=0
)
env = gym.wrappers.FrameStack(env, num_stack=4)

test_agent = DDQNPerAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=True,
    config=config,
)
test_agent.policy_network.load_state_dict(torch.load(save_model_name))

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
    print(f"{i_episode} episode Total Reward: {test_reward}")
env.close()
