import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        # defining the layers
        # fc1 is the transformation from state (input) to hidden layer
        # fc2 is the transformation from hidden layer to action (output)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # forward pass through the network
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x