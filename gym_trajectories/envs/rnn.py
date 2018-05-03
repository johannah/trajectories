# from KK
from copy import deepcopy
import time
import os
import torch 
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil

torch.manual_seed(139)

class RNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTMCell(1, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, xt, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(xt, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t

def train(e,do_save=False):
    optim.zero_grad()
    h1_tm1 = Variable(torch.FloatTensor(batch_size, hidden_size), requires_grad=False)*0.0
    c1_tm1 = Variable(torch.FloatTensor(batch_size, hidden_size), requires_grad=False)*0.0
    h2_tm1 = Variable(torch.FloatTensor(batch_size, hidden_size), requires_grad=False)*0.0
    c2_tm1 = Variable(torch.FloatTensor(batch_size, hidden_size), requires_grad=False)*0.0
    outputs = []
    # one batch of x
    for i in range(len(x)):
        output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
        outputs+=[output]
    y_pred = torch.stack(outputs, 0)
    mse_loss = ((y_pred-y)**2).mean()
    mse_loss.backward()
    clip = 10
    for p in rnn.parameters():
        p.grad.data.clamp_(min=-clip,max=clip)

    optim.step()
    if not e%100:
        ll = mse_loss.cpu().data.numpy()
        print('saving epoch {} loss {}'.format(e,ll))
        if np.isnan(ll[0]):
            embed()
        state = {'epoch':e, 
                'loss':ll,
                'state_dict':rnn.state_dict(), 
                'optimizer':optim.state_dict(), 
                 }
        filename = os.path.join(savedir, 'model_epoch_%06d.pkl'%e)
        save_checkpoint(state, filename=filename)
    return y_pred

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of {}".format(filename))
    f = open(filename, 'w')
    torch.save(state, f)
    f.close()
    print("finishing save of {}".format(filename))


def train_dummy_data():
    input_size, hidden_size, output_size = 1,128,1
    seq_length = 20
    lr = 1e-4
    
    # make sine wave data
    data_time_steps = np.linspace(2,10, seq_length+1)
    data = np.sin(data_time_steps)
    data.resize((seq_length+1), 1)
    batch_size = 10
    batch_data = np.array([data for d in range(batch_size)]).transpose(1,0,2)
    
    # target is input data shifted by one time step
    # input data should be - timestep, batchsize, features!
    x = Variable(torch.FloatTensor(batch_data[:-1]), requires_grad=False)
    y = Variable(torch.FloatTensor(batch_data[1:]), requires_grad=False)
    
    rnn = RNN(hidden_size=hidden_size)
    optim = optim.Adam(rnn.parameters(), lr=lr)
    best_loss = 7000
    savedir = 'saved'
    if not os.path.exists(savedir):
       os.makedirs(savedir)
    
    for e in range(4000):
        if not e%10:
            y_pred = train(e,do_save=True)
        else:
            y_pred = train(e,do_save=False)
    plt.plot(y_pred.data.numpy()[:,0], label='ypred')
    plt.plot(y.data.numpy()[:,0], label='y')
    plt.legend()
    plt.show()
    embed()

def train_vae_data():

if __name__ == '__main__':
    input_size, hidden_size, output_size = 1,128,1
    seq_length = 20
    lr = 1e-4
    
    # make sine wave data
    data_time_steps = np.linspace(2,10, seq_length+1)
    data = np.sin(data_time_steps)
    data.resize((seq_length+1), 1)
    batch_size = 10
    batch_data = np.array([data for d in range(batch_size)]).transpose(1,0,2)
    
    # target is input data shifted by one time step
    # input data should be - timestep, batchsize, features!
    x = Variable(torch.FloatTensor(batch_data[:-1]), requires_grad=False)
    y = Variable(torch.FloatTensor(batch_data[1:]), requires_grad=False)
    
    rnn = RNN(hidden_size=hidden_size)
    optim = optim.Adam(rnn.parameters(), lr=lr)
    best_loss = 7000
    savedir = 'saved'
    if not os.path.exists(savedir):
       os.makedirs(savedir)
    
    for e in range(4000):
        if not e%10:
            y_pred = train(e,do_save=True)
        else:
            y_pred = train(e,do_save=False)
    plt.plot(y_pred.data.numpy()[:,0], label='ypred')
    plt.plot(y.data.numpy()[:,0], label='y')
    plt.legend()
    plt.show()
    embed()

