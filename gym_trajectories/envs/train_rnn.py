# from KK
from torchnet.logger import VisdomPlotLogger, VisdomLogger
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
from datasets import EpisodicFroggerDataset, EpisodicDiffFroggerDataset

from glob import glob
from vq_vae_small import AutoEncoder, to_scalar
from conv_vae import Encoder, Decoder, VAE
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
from utils import get_cuts
#worst_inds = np.load('worst_inds.npz')['arr_0']
#all_inds = range(800)
#best_inds = np.array([w for w in all_inds if w not in list(worst_inds)])

#pcad = np.load('pca_components_vae.npz')
#V = pcad['V']
#vae_mu_mean = pcad['Xmean']
#vae_mu_std = pcad['Xstd']
#vae_Xpca_std = pcad['Xpca_std']
torch.manual_seed(139)

def test(e,dataloader,window_size,do_use_cuda=False):
    losses = []
    for batch_idx, (data_mu_diff_scaled, _, _, data_sigma_diff_scaled, _, _, name) in enumerate(dataloader):
        batch_losses = []
        batch_size = data_mu_diff_scaled.shape[0]
        # only use relevant mus
        # data_mu_scaled is example, timesteps, features
        # data shoud be timestep,batchsize,features
        data = data_mu_diff_scaled.permute(1,0,2)
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
        clip = 10
        cuts = get_cuts(x.shape[0], window_size)
        for st,en in cuts:
            for i in range(st, en):
                #print("forward i", i)
                output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
                outputs+=[output]
            local_pred = torch.stack(outputs[st:en], 0)
            mse_loss = ((local_pred-y[st:en])**2).mean()
            ll = mse_loss.cpu().data.numpy()[0]
            batch_losses.append(ll)
        losses.extend(batch_losses)
        pred = torch.stack(outputs, 0)
    return losses

def train(e,dataloader,window_size,do_use_cuda=False):
    losses = []
    cnt = 0
    for batch_idx, (data_mu_diff_scaled, _, _, data_sigma_diff_scaled, _, _, name) in enumerate(dataloader):
        if not batch_idx % 10:
            print('epoch {} batch_idx {}'.format(e,batch_idx))
        batch_losses = []
        batch_size = data_mu_diff_scaled.shape[0]
        # only use relevant mus
        # data_mu_scaled is example, timesteps, features
        # data shoud be timestep,batchsize,features
        data = data_mu_diff_scaled.permute(1,0,2)
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
        clip = 10
        cnt +=batch_size

        cuts = get_cuts(x.shape[0], window_size)
        for st,en in cuts:
            for i in range(st, en):
                #print("forward i", i)
                output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
                outputs+=[output]
            # truncated backprop
            optim.zero_grad()
            #print('backprop', st, en)
            local_pred = torch.stack(outputs[st:en], 0)
            mse_loss = ((local_pred-y[st:en])**2).mean()
            mse_loss.backward()
            for p in rnn.parameters():
                p.grad.data.clamp_(min=-clip,max=clip)
            optim.step()
            # detach hiddens and output
            h1_tm1 = h1_tm1.detach()
            c1_tm1 = c1_tm1.detach()
            h2_tm1 = h2_tm1.detach()
            c2_tm1 = c2_tm1.detach()
            ll = mse_loss.cpu().data.numpy()[0]
            batch_losses.append(ll)
        losses.extend(batch_losses)
        pred = torch.stack(outputs, 0)
    return cnt, losses

def save_checkpoint(state, filename='model.pkl'):
    print("starting save of {}".format(filename))
    f = open(filename, 'w')
    torch.save(state, f)
    f.close()
    print("finishing save of {}".format(filename))


def get_pca(dataloader):
    from scipy import linalg
    all_data = np.empty((0,50))
    for batch_idx, (data_mu, data_sigma, name) in enumerate(dataloader):
        rs = data_mu.numpy().reshape(data_mu.shape[0]*169,800)[:,best_inds]
        all_data = np.vstack((all_data, rs))

    X = all_data
    Xmean = np.mean(X, axis=0)
    Xstd = np.std(X, axis=0)
    X -= Xmean
    U, S, V = linalg.svd(X, full_matrices=False)
    Xpca = np.dot(X, V.T)
    Xpca_std = np.std(Xpca, axis=0)
    np.savez('pca_components_vae.npz', V=V, Xmean=Xmean, Xstd=Xstd, Xpca_std=Xpca_std)

    # to transform
    # X_transformed = np.dot(X-mean, V.T)/pca_std
    # to remove the transform
    # np.dot(X_transformed*pca_td, V) + mean
    # add mean back in

def pca_transform(data):
    return np.dot(data, V.T)-vae_mu_mean

def pca_untransform(data):
    return np.dot(data, V)+vae_mu_mean

if __name__ == '__main__':
    import argparse
    default_base_datadir = '/localdata/jhansen/trajectories_frames/dataset/'
    default_base_savedir = '/localdata/jhansen/trajectories_frames/saved/'
    parser = argparse.ArgumentParser(description='train for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-t', '--transform', default='std')
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-l', '--rnn_model_loadpath', default=None)
    parser.add_argument('-s', '--savename', default='base')
    parser.add_argument('-dt', '--data_type', default='diff')
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-hs', '--hidden_size', default=512, type=int)
    parser.add_argument('-se', '--save_every', default=10, type=int)
    parser.add_argument('-p', '--plot_port', default=8097, type=int)
    parser.add_argument('-w', '--window_size', default=10, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-ds', '--dataset_name', default='results')


    args = parser.parse_args()
    port = args.plot_port
    print("plotting to port %s" %port)
    print("make sure visdom server is running")
    print("python -m visdom.server -port %s" %port)


    basename = 'rnn_n%s_dt_%s_hs%04d_ws%03d_transform%s_%s' %(args.savename,
                                      args.data_type,
                                      args.hidden_size,
                                      args.window_size,
                                      args.transform,
                                      args.dataset_name,
                                      )


    train_loss_logger = VisdomPlotLogger(
              'line', port=port, opts={'title': '%s - Train Loss'%basename})

    test_loss_logger = VisdomPlotLogger(
              'line', port=port, opts={'title': '%s - Test Loss'%basename})
    use_cuda = args.cuda
    hidden_size = args.hidden_size
    # input after only good parts of vae taken
    input_size = 50
    lr = 1e-4
    rnn = RNN(input_size,hidden_size)
    optim = optim.Adam(rnn.parameters(), lr=lr, weight_decay=1e-6)
    if use_cuda:
        rnn.cuda()
    rnn_epoch = 0
    total_passes = 0

    train_loss = []
    test_loss = []
    if args.rnn_model_loadpath is not None:
        if  os.path.exists(args.rnn_model_loadpath):
            rnn_model_dict = torch.load(args.rnn_model_loadpath)
            rnn.load_state_dict(rnn_model_dict['state_dict'])
            optim.load_state_dict(rnn_model_dict['optimizer'])
            rnn_epoch = rnn_model_dict['epoch']
            try:
                total_passes = rnn_model_dict['total_passes']
                train_loss = rnn_model_dict['train_loss']
                test_loss = rnn_model_dict['test_loss']
            except:
                print("could not load total passes")
            print("loaded rnn from %s at epoch %s" %(args.rnn_model_loadpath, rnn_epoch))
        else:
            print("could not find model at %s"%args.rnn_model_loadpath)
            sys.exit()
    else:
        print("creating new model")

    if args.dataset_name == 'results':
        test_data_name = 'episodic_vae_test_results/'
    if args.dataset_name == 'small':
        test_data_name = 'episodic_vae_test_small/'
    if args.dataset_name == 'dummy':
        test_data_name =  'episodic_vae_test_dummy/'
    if args.dataset_name == 'tiny':
        test_data_name =  'episodic_vae_test_tiny/'

    print("using dataset: %s" %test_data_name)
    train_data_name = test_data_name.replace('test', 'train')
    test_data_path =  os.path.join(args.datadir,test_data_name)
    train_data_path = os.path.join(args.datadir,train_data_name)

    if args.data_type == 'diff':
        test_data_loader = DataLoader(EpisodicDiffFroggerDataset(test_data_path, transform=args.transform), batch_size=32, shuffle=True)
        train_data_loader = DataLoader(EpisodicDiffFroggerDataset(train_data_path, transform=args.transform, limit=args.num_train_limit), batch_size=32, shuffle=True)
    else:
        test_data_loader = DataLoader(EpisodicFroggerDataset(test_data_path, transform=args.transform), batch_size=32, shuffle=True)
        train_data_loader = DataLoader(EpisodicFroggerDataset(train_data_path, transform=args.transform, limit=args.num_train_limit), batch_size=32, shuffle=True)


    print("starting training")
    for e in range(rnn_epoch+1,rnn_epoch+args.num_epochs):
        ep_cnt, train_l = train(e,train_data_loader,args.window_size,do_use_cuda=use_cuda)
        total_passes +=ep_cnt
        test_l = test(e,test_data_loader,args.window_size,do_use_cuda=use_cuda)

        train_loss.append(np.mean(train_l))
        test_loss.append(np.mean(test_l))
        train_loss_logger.log(e,train_loss[-1])
        test_loss_logger.log(e, test_loss[-1])
        print('epoch {} train loss mean {} test loss mean {}'.format(e,
                              train_loss[-1],
                              test_loss[-1]))

        if ((not e%args.save_every) or (e == rnn_epoch+args.num_epochs)):
            state = {'epoch':e,
                    'train_loss':train_loss,
                    'test_loss':test_loss,
                    'state_dict':rnn.state_dict(),
                    'optimizer':optim.state_dict(),
                    'total_passes':total_passes,
                     }
            filename = os.path.join(default_base_savedir , basename + "e%05d.pkl"%e)
            save_checkpoint(state, filename=filename)
            time.sleep(5)


    e+=1
    #get_pca(train_data_loader)
    embed()
