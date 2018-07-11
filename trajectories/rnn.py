import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from IPython import embed

torch.manual_seed(139)

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        init.normal(self.lstm1.weight_ih,0.0,0.01)
        init.normal(self.lstm1.weight_hh,0.0,0.01)
        init.normal(self.lstm2.weight_ih,0.0,0.01)
        init.normal(self.lstm2.weight_hh,0.0,0.01)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_t, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(x_t, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t


