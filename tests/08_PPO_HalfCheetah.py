import gymnasium as gym
import torch

def main():
    args = {
        "env_name" : "HalfCheetah-v4",
        "ep_len" : 1000,
        "max_step" : 200000
    }

    env = gym.make(args["env_name"], render_mode="human")


if __name__ == "__main__":
    main()