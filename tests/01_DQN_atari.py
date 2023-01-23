# %%
import gymnasium as gym
import torch

print("gym version:[%s]" % (gym.__version__))
print("Pytorch:[%s]" % (torch.__version__))

from src.agents.dqn_agent import DQNAgent
from src.utils.util import print_env_info

#%%
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("ALE/Enduro-v5")
print_env_info(env=env)
print(env.observation_space.shape)
# %%
agent = DQNAgent(
    obs_space_dims=env.observation_space.shape[0],
    action_space_dims=env.action_space.n,
    is_atari=True,
)
print(agent.config)

# %%
for episode in range(agent.config.n_episodes):
    obs, info = env.reset()
    done = False
    ep_ret = 0
    ep_len = 0
    while not done:
        # env.render()
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)

        if len(agent.memory.buffer) > 2000:
            agent.update()

        obs = next_obs
        ep_ret += reward
        ep_len += 1

    agent.decay_epsilon()

    if (episode + 1) % agent.config.target_update:
        agent.q_target.load_state_dict(agent.q_predict.state_dict())

    if (episode == 0) or (((episode + 1) % 10) == 0):
        print(
            f"episode: {episode + 1} | ep_ret: {ep_ret:.4f} | epsilon: {agent.epsilon:.4f}"
        )

env.close()

# %%
