# from KK
from rnn import RNN
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

def train(e,dataloader,do_save=False,do_use_cuda=False):
    losses = []
    for batch_idx, (data_mu, data_sigma, name) in enumerate(dataloader):
        optim.zero_grad()
        batch_size = data_mu.shape[0]
        # only use relevant mus
        # data_mu is example, timesteps, features
        # data shoud be timestep,batchsize,features
        data_all = data_mu.permute(1,0,2)
        data = data_all[:,:,best_inds]
        if do_use_cuda:
            seq = Variable(torch.FloatTensor(data), requires_grad=False).cuda()
            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
        else:
            seq = Variable(torch.FloatTensor(data), requires_grad=False)
            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)

        y = seq[1:]
        x = seq[:-1]
        outputs = []
        for i in range(len(x)):
            output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            outputs+=[output]
        pred = torch.stack(outputs, 0)
        mse_loss = ((pred-y)**2).mean()
        mse_loss.backward()
        clip = 10
        for p in rnn.parameters():
            p.grad.data.clamp_(min=-clip,max=clip)
        optim.step()
        ll = mse_loss.cpu().data.numpy()[0]
        if np.isnan(ll):
            embed()
        losses.append(ll)
        if not batch_idx%100:
            print('epoch {} batch_idx {} loss {}'.format(e,batch_idx,ll))
    if do_save:
        print('saving epoch {} loss {}'.format(e,np.mean(losses)))
        state = {'epoch':e,
                'loss':np.mean(losses),
                'state_dict':rnn.state_dict(),
                'optimizer':optim.state_dict(),
                 }
        filename = os.path.join(default_base_savedir , '%s_rnn_model_epoch_%06d_loss%05f.pkl'%(args.savename,e,np.mean(losses)))
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
    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-l', '--rnn_model_loadpath', default=None)
    parser.add_argument('-s', '--savename', default='base')
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-se', '--save_every', default=100, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)


    args = parser.parse_args()
    use_cuda = args.cuda
    hidden_size = 512
    # input after only good parts of vae taken
    input_size = 50
    lr = 1e-4
    rnn = RNN(input_size,hidden_size)
    optim = optim.Adam(rnn.parameters(), lr=lr, weight_decay=1e-6)
    if use_cuda:
        rnn.cuda()
    rnn_epoch = 0
    if args.rnn_model_loadpath is not None:
        if  os.path.exists(args.rnn_model_loadpath):
            rnn_model_dict = torch.load(args.rnn_model_loadpath)
            rnn.load_state_dict(rnn_model_dict['state_dict'])
            optim.load_state_dict(rnn_model_dict['optimizer'])
            rnn_epoch = rnn_model_dict['epoch']
            print("loaded rnn from %s at epoch %s" %(args.rnn_model_loadpath, rnn_epoch))
        else:
            print("could not find model at %s"%args.rnn_model_loadpath)
            sys.exit()


    test_data_name = 'episodic_vae_test_results/'
    #test_data_name =  'episodic_vae_test_dummy/'
    #test_data_name =  'episodic_vae_test_tiny/'
    train_data_name = test_data_name.replace('test', 'train')

    test_data_path =  os.path.join(args.datadir,test_data_name)
    train_data_path = os.path.join(args.datadir,train_data_name)

    test_data_loader = DataLoader(EpisodicFroggerDataset(test_data_path), batch_size=32, shuffle=False)
    train_data_loader = DataLoader(EpisodicFroggerDataset(train_data_path, limit=args.num_train_limit), batch_size=32, shuffle=False)
    for e in range(rnn_epoch+1,rnn_epoch+args.num_epochs):
        if not e%args.save_every:
            mean_loss = train(e,train_data_loader,do_save=True,do_use_cuda=use_cuda)
        else:
            mean_loss = train(e,train_data_loader,do_save=False,do_use_cuda=use_cuda)

    e+=1
    mean_loss = train(e,train_data_loader,do_save=True,do_use_cuda=use_cuda)
    embed()
