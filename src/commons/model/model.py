import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        assert obs_space_dims > 0
        assert action_space_dims > 0

        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_space_dims, 24),
            nn.ReLU(),
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_dims),
        )

    def forward(self, x):
        return self.fc(x)


class CNNModel(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        assert obs_space_dims > 0
        assert action_space_dims > 0

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(obs_space_dims, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(), nn.Linear(512, action_space_dims)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def _init_weights(self, moodule):
        if isinstance(moodule, nn.Linear):
            torch.nn.init.xavier_uniform_(moodule.weight)
            moodule.bias.data.fill_(0.01)
