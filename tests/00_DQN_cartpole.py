# %%
import gymnasium as gym
import torch
from src.utils.util import ShellColor as sc

print(f"{sc.COLOR_PURPLE}Gym version:{sc.ENDC} {gym.__version__}")
print(f"{sc.COLOR_PURPLE}Pytorch version:{sc.ENDC} {torch.__version__}")

from src.agents.dqn_agent import DQNAgent
from src.utils.util import print_env_info
from src.utils.plot import plot_graph

#%%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")
print_env_info(env=env)
env.observation_space.shape
# %%
agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
)
agent.config.n_episodes = 1000
agent.config.target_update = 50
print(agent.config)

# %%
rewards = []
losses = []
for episode in range(agent.config.n_episodes):
    obs, info = env.reset()
    done = False
    ep_ret = 0
    ep_len = 0
    frame = 0
    avg_loss = 0
    while not done:
        # env.render()
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)

        if len(agent.memory.buffer) > 2000:
            agent.update()
            avg_loss += agent.loss

        obs = next_obs
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

plot_graph(rewards, ylabel="score")
plot_graph(losses, ylabel="loss")
# %%
torch.save(agent.q_predict.state_dict(), "checkpoint.pth")


# %%
# load the weights from file
env = gym.make("CartPole-v1", render_mode="human")
test_agent = DQNAgent(
    obs_space_shape=env.observation_space.shape,
    action_space_dims=env.action_space.n,
    is_atari=False,
)
test_agent.q_predict.load_state_dict(torch.load("checkpoint.pth"))

for i in range(1000):
    obs, info = env.reset()
    for j in range(100):
        action = test_agent.get_action(obs)
        env.render()
        next_obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            break

env.close()

# %%
