import gymnasium as gym

env = gym.make("ALE/Enduro-v5", render_mode="human")
observation, info = env.reset()
env.metadata["render_fps"] = 2
for _ in range(1000):
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    env.render()
env.close()
