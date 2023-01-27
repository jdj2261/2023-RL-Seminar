import cv2
import torch
import numpy as np
import gymnasium as gym
from dataclasses import dataclass


def create_config() -> dict:
    config = {}
    config["n_episodes"] = 10000
    config["batch_size"] = 64
    config["gamma"] = 0.99
    config["lr"] = 5e-4
    config["epsilon_start"] = 0.95
    config["epsilon_end"] = 0.01
    config["seed"] = 0
    config["memory_type"] = "uniform"
    config["memory_capacity"] = 100000
    return config


@dataclass
class Config:
    n_episodes = 1000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 0.0005
    epsilon_start: float = 0.95
    epsilon_end: float = 0.01
    seed: int = 0
    target_update: int = 5

    memory_type: str = "uniform"  # "priority"
    memory_capacity: int = 100000

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
            + f"{ShellColor.COLOR_CYAN}target_update:{ShellColor.ENDC} {self.target_update}"
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


def get_preprocessed_img(image):
    """
    Process image crop resize, grayscale and normalize the images
    """
    # print(image[0])
    preprocessed_img = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    # preprocessed_img = np.expand_dims(preprocessed_img, -1)
    # preprocessed_img = np.transpose(preprocessed_img, (2, 0, 1)).astype(np.float32)
    preprocessed_img = preprocessed_img.astype(np.float32) / 255
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
