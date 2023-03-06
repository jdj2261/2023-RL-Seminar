import torch
import torch.nn as nn

from src.agents.base_agent import Agent


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Initialize state, action dimension
        # state, action 차원을 초기화한다. 
        self.state_dim = state_dim
        self.action_dim = action_dim
        

class PPOAgent(Agent):
    def __init__(
            self,
            obs_space_shape: tuple,
            action_space_dims: int,
            config: dict
    ):
        super().__init__(obs_space_shape, action_space_dims, config)
