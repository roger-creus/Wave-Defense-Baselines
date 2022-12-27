import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import *
from utils.smirl import *

from IPython import embed

class Encoder(nn.Module):
    def __init__(self, in_channels = 4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        # in_channels is 4 bc the frame stack
        self.network = Encoder(in_channels = 4)

        self.fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.fc(self.network(x / 255.0)))

    def get_action_and_value(self, x, action=None):
        hidden = self.fc(self.network(x / 255.0))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
        
class Agent_SMIRL(nn.Module):
    def __init__(self, envs, latent_dim = 128):
        super().__init__()

        self.latent_dim = latent_dim
        
        self.vae = VAE(in_channels = 3, latent_dim = self.latent_dim)

        # input to smirl policy is z, mu(z), sigma(z) where mus and sigmas are computed episodic-like
        self.actor = layer_init(nn.Linear(self.latent_dim * 3, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(self.latent_dim * 3, 1), std=1)

    def get_value(self, x, ep_mu, ep_sigma):
        mu, logvar = self.vae.encode(x / 255.0)
        z = self.vae.reparameterize(mu, logvar)

        x = torch.cat([z, ep_mu, ep_sigma], axis = 1)
        
        return self.critic(x)


    def get_action_and_value(self, x, ep_mu, ep_sigma, action=None):
        mu, logvar = self.vae.encode(x / 255.0)
        z = self.vae.reparameterize(mu, logvar)

        x = torch.cat([z, ep_mu, ep_sigma], axis = 1)

        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class Agent_RND(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )

        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.fc(self.network(x / 255.0))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.fc(self.network(x / 255.0))
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class Agent_MLP(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x = self.shared(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.shared(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
