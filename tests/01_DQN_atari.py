# %%
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import wandb
from collections import deque
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils import util as rl_util

#%%
env_name = "PongNoFrameskip-v4"
env = gym.make(env_name)
env = gym.wrappers.AtariPreprocessing(
    env=env, terminal_on_life_loss=True, grayscale_obs=True, scale_obs=False, noop_max=0
)
# env = ImageToPyTorch(env)
env = gym.wrappers.FrameStack(env, num_stack=4)
rl_util.print_env_info(env=env)


# %%
config = rl_util.create_config()
config["seed"] = 42
wandb.init(project="DDQN-Atari", config=config)

#%%
agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=True,
    config=config,
)

print(agent.config)
print(type(agent.memory))

#%%
save_dir = "result/DDQN/atari/"
rl_util.create_directory(save_dir)
save_model_name = save_dir + env_name + "_mean_score.pt"
# %%
state, _ = env.reset()
episode_rewards = []
episode_reward = 0
num_episodes = 0
best_mean_reward = -10000

for t in range(agent.config.max_steps):
    fraction = min(1.0, float(t) / agent.config.epsilon_decay)
    eps_threshold = agent.config.epsilon_start + fraction * (
        agent.config.epsilon_end - agent.config.epsilon_start
    )
    # sample = random.random()
    action = agent.select_action(state, eps_threshold)
    next_state, reward, done, _, info = env.step(action)
    agent.store_transition(state, action, reward, next_state, float(done))
    state = next_state

    episode_reward += reward
    if done:
        state, _ = env.reset()
        episode_rewards.append(episode_reward)
        episode_reward = 0

    if (
        t > agent.config.start_training_step
        and t % agent.config.learning_frequency == 0
    ):
        agent.update()

    if (
        t > agent.config.start_training_step
        and t % agent.config.target_update_frequency == 0
    ):
        agent.update_target_network()

    num_episodes = len(episode_rewards)

    if (
        done
        and agent.config.print_frequency is not None
        and len(episode_rewards) % agent.config.print_frequency == 0
    ):
        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        print("********************************************************")
        print("steps: {}".format(t))
        print("episodes: {}".format(num_episodes))
        print("mean 100 episode reward: {}".format(mean_100ep_reward))
        print("% epsilon: {}".format(eps_threshold))
        print("********************************************************")
        wandb.log(
            {
                "episode": num_episodes,
                "mean reward": mean_100ep_reward,
                "epsilon": eps_threshold,
            }
        )
        if best_mean_reward < mean_100ep_reward:
            torch.save(agent.policy_network.state_dict(), save_model_name)
            print(
                f"Best mean reward updated {best_mean_reward:.3f} -> {mean_100ep_reward:.3f}, model saved"
            )
            best_mean_reward = mean_100ep_reward
            if mean_100ep_reward > agent.config.mean_reward_bound:
                print(f"Solved!")
                break
            else:
                if num_episodes > 1000:
                    break

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
