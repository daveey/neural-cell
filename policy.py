import pufferlib.vectorization
import pufferlib.emulation
import gym
import pufferlib.frameworks.cleanrl
import torch
from torch import nn
import numpy as np

class Policy(nn.Module):
    def __init__(self, env: pufferlib.emulation.GymPufferEnv):
        super().__init__()
        self.encoder = nn.Linear(np.prod(
            env.observation_space.shape), 128)
        self.decoders = nn.ModuleList([nn.Linear(128, n)
            for n in env.observation_space.shape])
        self.value_head = nn.Linear(128, 1)

    def forward(self, env_outputs):
        env_outputs = env_outputs.reshape(env_outputs.shape[0], -1)
        hidden = self.encoder(env_outputs)
        actions = [dec(hidden) for dec in self.decoders]
        value = self.value_head(hidden)
        return actions, value
