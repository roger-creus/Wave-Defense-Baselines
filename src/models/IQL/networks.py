import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, in_channels, action_size, hidden_size=512):
        super(Actor, self).__init__()
       
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.conv(state)
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits
    
    def evaluate(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return action, dist
        
    def get_action(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.detach().cpu()

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, in_channels, action_size, hidden_size=512, seed=1):
        super(Critic, self).__init__()
        torch.manual_seed(seed)
        
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Value(nn.Module):
    """Value (Value) Model."""

    def __init__(self, in_channels, hidden_size=512):
        super(Value, self).__init__()
        
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(64 * 7 * 7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.conv(state)
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)