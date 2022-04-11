#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-23 15:29:24
LastEditor: John
LastEditTime: 2021-04-08 22:36:43
Discription:
Environment:
'''
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch
import torch.nn.functional as F
from torch.distributions import Beta, Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_head = nn.Linear(hidden_dim, action_dim)


    def forward(self, state):
        out_put = self.actor(state)
        mu = torch.sigmoid(self.mu_head(out_put))
        sigma = F.softplus(self.sigma_head(out_put))
        dist = Normal(mu, sigma)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value
