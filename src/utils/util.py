import cv2
import torch
import numpy as np
import os
import gymnasium as gym
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime


def create_config() -> dict:
    config = {}
    config["n_episodes"] = 10000
    config["batch_size"] = 64
    config["gamma"] = 0.99
    config["lr"] = 1e-4
    config["epsilon_start"] = 0.95
    config["epsilon_end"] = 0.01
    config["seed"] = 0
    config["update_frequency"] = 4
    config["memory_type"] = "uniform"
    config["memory_capacity"] = 1000
    config["device"] = "cpu"
    return config


@dataclass
class Config:
    n_episodes = 10000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 0.001
    epsilon_start: float = 0.95
    epsilon_end: float = 0.01
    seed: int = 0
    update_frequency: int = 4
    memory_type: str = "uniform"  # "priority"
    memory_capacity: int = 1000
    device: str = "cpu"

    def __repr__(self) -> str:
        result = "=" * 10 + " Config Info " + "=" * 10
        result += (
            "\n"
            + f"{ShellColor.COLOR_CYAN}n_episodes:{ShellColor.ENDC} {self.n_episodes}"
        )
        result += (
            "\n"
            + f"{ShellColor.COLOR_CYAN}batch_size:{ShellColor.ENDC} {self.batch_size}"
        )
        result += "\n" + f"{ShellColor.COLOR_CYAN}gamma:{ShellColor.ENDC} {self.gamma}"
        result += "\n" + f"{ShellColor.COLOR_CYAN}lr:{ShellColor.ENDC} {self.lr}"
        result += (
            "\n"
            + f"{ShellColor.COLOR_CYAN}epsilon_start:{ShellColor.ENDC} {self.epsilon_start}"
        )
        result += (
            "\n"
            + f"{ShellColor.COLOR_CYAN}epsilon_end:{ShellColor.ENDC} {self.epsilon_end}"
        )
        result += "\n" + f"{ShellColor.COLOR_CYAN}seed:{ShellColor.ENDC} {self.seed}"
        result += (
            "\n"
            + f"{ShellColor.COLOR_CYAN}update_frequency:{ShellColor.ENDC} {self.update_frequency}"
        )
        result += (
            "\n"
            + f"{ShellColor.COLOR_CYAN}memory_type:{ShellColor.ENDC} {self.memory_type}"
        )
        result += (
            "\n"
            + f"{ShellColor.COLOR_CYAN}memory_capacity:{ShellColor.ENDC} {self.memory_capacity}"
        )
        result += (
            "\n" + f"{ShellColor.COLOR_CYAN}device:{ShellColor.ENDC} {self.device}"
        )
        result += "\n" + "=" * 33
        return f"{result}"


class ShellColor:
    COLOR_NC = "\033[0m"  # No Color
    COLOR_BLACK = "\033[0;30m"
    COLOR_GRAY = "\033[1;30m"
    COLOR_RED = "\033[0;31m"
    COLOR_LIGHT_RED = "\033[1;31m"
    COLOR_GREEN = "\033[0;32m"
    COLOR_LIGHT_GREEN = "\033[1;32m"
    COLOR_BROWN = "\033[0;33m"
    COLOR_YELLOW = "\033[1;33m"
    COLOR_BLUE = "\033[0;34m"
    COLOR_LIGHT_BLUE = "\033[1;34m"
    COLOR_PURPLE = "\033[0;35m"
    COLOR_LIGHT_PURPLE = "\033[1;35m"
    COLOR_CYAN = "\033[0;36m"
    COLOR_LIGHT_CYAN = "\033[1;36m"
    COLOR_LIGHT_GRAY = "\033[0;37m"
    COLOR_WHITE = "\033[1;37m"

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def get_preprocessed_img(image, heigt, width):
    """
    Process image crop resize, grayscale and normalize the images
    """
    # print(image[0])
    preprocessed_img = cv2.cvtColor(
        cv2.resize(image, (heigt, width), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    # preprocessed_img = np.expand_dims(preprocessed_img, -1)
    # preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1)).astype(np.float32)
    preprocessed_img = preprocessed_img.astype(np.uint8)
    return preprocessed_img


def print_env_info(env: gym.Env) -> None:
    print("=" * 12 + " Env Info " + "=" * 11)
    print(f"{ShellColor.COLOR_CYAN}Env id:{ShellColor.ENDC} {env.spec.id}")
    print(
        f"{ShellColor.COLOR_CYAN}Observation space size:{ShellColor.ENDC} {env.observation_space.shape}"
    )
    print(
        f"{ShellColor.COLOR_CYAN}Action space size:{ShellColor.ENDC} {env.env.action_space.n}"
    )

    if "CartPole" not in env.env.spec.id:
        print(
            f"{ShellColor.COLOR_CYAN}Action list:{ShellColor.ENDC} {env.env.get_action_meanings()}"
        )
    print("=" * 33)


def cartpole_evaluate_agent(env, agent, num=10):
    reward_sum = 0.0
    for _ in range(num):
        obs, _ = env.reset()
        while True:
            state = torch.tensor(obs, dtype=torch.float, device=agent.config.device)
            with torch.no_grad():
                action = torch.argmax(agent.q_predict(state))
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward_sum += reward

            obs = next_obs
            if done:
                break
    return reward_sum / float(num)


def atari_evaluate_agent(env, agent, num=10):
    reward_sum = 0.0
    for _ in range(num):
        obs, _ = env.reset()
        obs = np.squeeze(obs, axis=-1)
        while True:
            env.render()
            state = torch.tensor(
                obs, dtype=torch.float, device=agent.config.device
            ).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(agent.q_predict(state))
            print(action)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            next_obs = np.squeeze(next_obs, axis=-1)
            done = terminated or truncated
            reward_sum += reward
            obs = next_obs
            if done:
                break
    return reward_sum / float(num)


def create_directory(dir_name):
    try:
        if not os.path.exists(dir_name):
            print(f"Create {dir_name} direcoty")
            os.makedirs(dir_name)
    except OSError:
        print("Error: Failed to create the directory.")


def get_current_time_string():
    return datetime.now().strftime("%Y_%m_%d_%I_%M_%S")


def init_2d_figure(name=None, figsize=(15, 7.5), dpi=80):
    fig = plt.figure(name, figsize=figsize, dpi=dpi)
    ax = plt.gca()
    return fig, ax


def plot_graph(
    ax,
    values,
    title="result",
    xlabel="episode",
    ylabel="score",
    is_save=False,
    save_dir_name="result/",
):
    ax.plot(values, label=ylabel)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.legend(prop={"size": 12})
    if is_save:
        create_directory(save_dir_name)
        file_name = save_dir_name + title + "_{}.png".format(get_current_time_string())
        print(f"Save {file_name}")
        plt.savefig(file_name)


def show_figure():
    """
    Show figure
    """
    plt.show()
