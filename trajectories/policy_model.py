import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

class MCTSNetwork(nn.Module):
    def __init__(self, input_size, action_space):
        super(MCTSNetwork, self).__init__()
        # input of state (s), output of move probabilities 
        # p with components p_a=Pr(a|s)
        # and scalar value v estimating expected outcome z from position s
        # loss function l =(z-v)**2 - pi log(p) + c||theta||**2
        # (p,v) = f_theta(s) 
        # c controls the level of L2 regularization
        self.layer_one = nn.Linear(input_size, 512)
        self.layer_two = nn.Linear(512, 256)
        self.layer_three = nn.Linear(256, 128)

        self.action_layer = nn.Linear(128, 64) 
        self.action_head = nn.Linear(64, action_space)

        self.value_layer = nn.Linear(128, 64) 
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.layer_one(x))
        x = F.relu(self.layer_two(x))
        x = F.relu(self.layer_three(x))

        ax = F.relu(self.action_layer(x))
        vx = F.relu(self.value_layer(x))

        action_scores = self.action_head(ax)
        state_values = self.value_head(vx)
        return F.softmax(action_scores, dim=-1), state_values

