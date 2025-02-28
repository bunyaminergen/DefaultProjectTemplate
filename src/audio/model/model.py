import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


def build_model(config):
    input_dim = config.get("input_dim", 784)
    hidden_dim = config.get("hidden_dim", 256)
    output_dim = config.get("output_dim", 10)

    model = MyModel(input_dim, hidden_dim, output_dim)

    return model
