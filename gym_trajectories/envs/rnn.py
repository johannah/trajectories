# from KK
from copy import deepcopy
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from IPython import embed
import shutil
from datasets import EpisodicFroggerDataset

from glob import glob
from vq_vae_small import AutoEncoder, to_scalar
from conv_vae import Encoder, Decoder, VAE
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
worst_inds = np.load('worst_inds.npz')['arr_0']
all_inds = range(800)
best_inds = np.array([w for w in all_inds if w not in list(worst_inds)])

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

    def forward(self, xt, h1_tm1, c1_tm1, h2_tm1, c2_tm1):
        h1_t, c1_t = self.lstm1(xt, (h1_tm1, c1_tm1))
        h2_t, c2_t = self.lstm2(h1_t, (h2_tm1, c2_tm1))
        output = self.linear(h2_t)
        return output, h1_t, c1_t, h2_t, c2_t

def train(e,dataloader,do_save=False,do_use_cuda=False):
    losses = []
    for batch_idx, (data_mu, data_sigma, name) in enumerate(dataloader):
        optim.zero_grad()
        batch_size = data_mu.shape[0]
        # only use relevant mus
        # data shoud be timestep,batchsize,features
        data = data_mu[:,:,best_inds].permute(1,0,2)
        if do_use_cuda:
            x = Variable(torch.FloatTensor(data), requires_grad=False).cuda()
            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
        else:
            x = Variable(torch.FloatTensor(data), requires_grad=False)
            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
        outputs = []
        for i in range(len(x)):
            output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            outputs+=[output]
        pred = torch.stack(outputs, 0)
        mse_loss = ((pred-x)**2).mean()
        mse_loss.backward()
        clip = 10
        for p in rnn.parameters():
            p.grad.data.clamp_(min=-clip,max=clip)
        optim.step()
        ll = mse_loss.cpu().data.numpy()[0]
        if np.isnan(ll):
            embed()
        losses.append(ll)
        if not batch_idx%10:
            print('epoch {} batch_idx {} loss {}'.format(e,batch_idx,ll))
    if do_save:
        print('saving epoch {} loss {}'.format(e,np.mean(losses)))
        state = {'epoch':e,
                'loss':np.mean(losses),
                'state_dict':rnn.state_dict(),
                'optimizer':optim.state_dict(),
                 }
        filename = os.path.join(default_base_savedir , 'rnn_model_epoch_%06d.pkl'%e)
        save_checkpoint(state, filename=filename)
        time.sleep(5)

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of {}".format(filename))
    f = open(filename, 'w')
    torch.save(state, f)
    f.close()
    print("finishing save of {}".format(filename))


if __name__ == '__main__':
    import argparse
    default_base_datadir = '/localdata/jhansen/trajectories_frames/dataset/'
    default_base_savedir = '/localdata/jhansen/trajectories_frames/saved/'
    default_model_savepath = os.path.join(default_base_savedir, 'conv_vae_model.pkl')
    default_rnn_model_savepath = os.path.join(default_base_savedir, 'rnn_vae.pkl')
    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-s', '--model_savepath', default=default_model_savepath)
    parser.add_argument('-l', '--model_loadpath', default=default_model_savepath)

    parser.add_argument('-rs', '--rnn_model_savepath', default=default_rnn_model_savepath)
    parser.add_argument('-rl', '--rnn_model_loadpath', default=default_rnn_model_savepath)

    parser.add_argument('-z', '--num_z', default=32, type=int)
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')

    args = parser.parse_args()
    use_cuda = args.cuda

    #dsize = 40
    #nr_mix = 10
    ## mean and scale for each components and weighting bt components (10+2*10)
    #probs_size = (2*nr_mix)+nr_mix
    #latent_size = 32

    #encoder = Encoder(latent_size)
    #decoder = Decoder(latent_size, probs_size)
    #vae = VAE(encoder, decoder, use_cuda)
    ## square error is not the correct loss - for ordered input,
    ## should use softmax for unordered input ( like mine )

    #if use_cuda:
    #    print("using gpu")
    #    vae = vae.cuda()
    #    vae.encoder = vae.encoder.cuda()
    #    vae.decoder = vae.decoder.cuda()
    #opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    #epoch = 0
    #if args.model_loadpath is not None:
    #    if os.path.exists(args.model_loadpath):
    #        model_dict = torch.load(args.model_loadpath)
    #        vae.load_state_dict(model_dict['state_dict'])
    #        opt.load_state_dict(model_dict['optimizer'])
    #        epoch =  model_dict['epoch']
    #        print('loaded checkpoint at epoch: {} from {}'.format(epoch, args.model_loadpath))
    #    else:
    #        print('could not find checkpoint at {}'.format(args.model_loadpath))
    #        embed()

    test_data_path =  os.path.join(args.datadir,'episodic_vae_test_results/')
    train_data_path = os.path.join(args.datadir,'episodic_vae_train_results/')
    test_data_loader = DataLoader(EpisodicFroggerDataset(test_data_path), batch_size=32, shuffle=True)
    train_data_loader = DataLoader(EpisodicFroggerDataset(train_data_path, limit=args.num_train_limit), batch_size=32, shuffle=True)
    hidden_size = 128
    # input after only good parts of vae taken
    input_size = 50
    seq_length = 169
    lr = 1e-4
    rnn = RNN(input_size,hidden_size)
    if use_cuda:
        rnn.cuda()

    optim = optim.Adam(rnn.parameters(), lr=lr, weight_decay=1e-6)
    for e in range(args.num_epochs):
        train(e,train_data_loader,do_save=True,do_use_cuda=use_cuda)
    embed()



