import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, obs_space_shape: tuple, action_space_dims: int):
        assert len(obs_space_shape) > 0
        assert action_space_dims > 0

        obs_space_dims = obs_space_shape[0]
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
    def __init__(self, input_shape: tuple, action_space_dims: int):
        super().__init__()
        assert action_space_dims > 0
        obs_space_dims = input_shape[0]
        h, w = input_shape[1], input_shape[2]

        print(input_shape)
        self.features = nn.Sequential(
            nn.Conv2d(obs_space_dims, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        convh, convw = self.conv2d_size_calc(h, w, kernel_size=8, stride=4)
        convh, convw = self.conv2d_size_calc(convh, convw, kernel_size=4, stride=2)
        convh, convw = self.conv2d_size_calc(convh, convw, kernel_size=3, stride=1)

        self.fc = nn.Sequential(
            nn.Linear(convh * convw * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_dims),
        )

        self.apply(self._init_weights)

    def conv2d_size_calc(self, h, w, kernel_size=1, stride=1, padding=0):
        """
        Calcs conv layers output image sizes
        """
        # (((W - K + 2P)/S) + 1)
        output_h = (h - kernel_size + 2 * padding) // stride + 1
        output_w = (w - kernel_size + 2 * padding) // stride + 1

        return output_h, output_w

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def _init_weights(self, moodule):
        if isinstance(moodule, nn.Linear):
            torch.nn.init.xavier_uniform_(moodule.weight)
            moodule.bias.data.fill_(0.01)
