import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, obs_space_shape: tuple, action_space_dims: int):
        assert len(obs_space_shape) > 0
        assert action_space_dims > 0

        obs_space_dims = obs_space_shape[0]
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_space_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_dims),
        )

    def forward(self, x):
        return self.fc(x)


class CNNModel(nn.Module):
    def __init__(self, input_shape: tuple, action_space_dims: int):
        super(CNNModel, self).__init__()

        print(f"Model Initializing... An input shape is {input_shape}")
        obs_space_dims = input_shape[0]
        self.conv_layer_1 = nn.Conv2d(obs_space_dims, 32, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.dense_layer = nn.Linear(7 * 7 * 64, 256)
        self.out_layer = nn.Linear(256, action_space_dims)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv_layer_1(x))
        x = F.relu(self.conv_layer_2(x))
        x = F.relu(self.conv_layer_3(x))
        x = F.relu(self.dense_layer(x.view(x.size(0), -1)))
        return self.out_layer(x)

class Dueling_CNNModel(nn.Module):
    def __init__(self, input_shape: tuple, action_space_dims: int):
        super(Dueling_CNNModel, self).__init__()

        print(f"Model Initializing... An input shape is {input_shape}")
        obs_space_dims = input_shape[0]
        self.num_actions = action_space_dims
        self.conv_layer_1 = nn.Conv2d(obs_space_dims, 32, kernel_size=8, stride=4)
        self.conv_layer_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv_layer_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=self.num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.out_layer = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv_layer_1(x))
        x = F.relu(self.conv_layer_2(x))
        x = F.relu(self.conv_layer_3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x