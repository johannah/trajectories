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
from datasets import EpisodicFroggerDataset

from glob import glob
from vq_vae_small import AutoEncoder, to_scalar
from conv_vae import Encoder, Decoder, VAE
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
#worst_inds = np.load('worst_inds.npz')['arr_0']
#all_inds = range(800)
#best_inds = np.array([w for w in all_inds if w not in list(worst_inds)])

#pcad = np.load('pca_components_vae.npz')
#V = pcad['V']
#vae_mu_mean = pcad['Xmean']
#vae_mu_std = pcad['Xstd']
#vae_Xpca_std = pcad['Xpca_std']
torch.manual_seed(139)


def test(e,dataloader,do_use_cuda=False):
    losses = []
    for batch_idx, (data_mu_scaled, _, data_sigma, name) in enumerate(dataloader):
        batch_losses = []
        batch_size = data_mu_scaled.shape[0]
        # only use relevant mus
        # data_mu_scaled is example, timesteps, features
        # data shoud be timestep,batchsize,features
        data = data_mu_scaled.permute(1,0,2)
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
        window_size = 20
        seen = 0
        for i in range(len(x)):
            output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            outputs+=[output]
            seen +=1
            if ((not i%window_size) and (seen>1) or (i==(len(x)-1))):
                optim.zero_grad()
                local_pred = torch.stack(outputs[i-(seen-1):i], 0)
                mse_loss = ((local_pred-y[i-(seen-1):i])**2).mean()
                seen = 0
                ll = mse_loss.cpu().data.numpy()[0]
                batch_losses.append(ll)
        losses.extend(batch_losses)
    return losses



def train(e,dataloader,do_use_cuda=False):
    losses = []
    cnt = 0
    for batch_idx, (data_mu_scaled, _, data_sigma, name) in enumerate(dataloader):
        batch_losses = []
        batch_size = data_mu_scaled.shape[0]
        # only use relevant mus
        # data_mu_scaled is example, timesteps, features
        # data shoud be timestep,batchsize,features
        data = data_mu_scaled.permute(1,0,2)
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
        window_size = 20
        seen = 0
        clip = 10
        cnt +=batch_size
        for i in range(len(x)):
            output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            outputs+=[output]
            seen +=1

            if ((not i%window_size) and (seen>1) or (i==(len(x)-1))):
                optim.zero_grad()
                local_pred = torch.stack(outputs[i-(seen-1):i], 0)
                mse_loss = ((local_pred-y[i-(seen-1):i])**2).mean()
                seen = 0
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
        if not batch_idx%100:
            print('epoch {} batch_idx {} loss {}'.format(e,batch_idx,np.mean((batch_losses))))
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
    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-t', '--transform', default='pca')
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-l', '--rnn_model_loadpath', default=None)
    parser.add_argument('-s', '--savename', default='base')
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-hs', '--hidden_size', default=512, type=int)
    parser.add_argument('-se', '--save_every', default=10, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)


    args = parser.parse_args()
    port = 8097
    train_loss_logger = VisdomPlotLogger(
              'line', port=port, opts={'title': '%s - Train Loss'%args.savename})

    test_loss_logger = VisdomPlotLogger(
              'line', port=port, opts={'title': '%s - Test Loss'%args.savename})
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
    if args.rnn_model_loadpath is not None:
        if  os.path.exists(args.rnn_model_loadpath):
            rnn_model_dict = torch.load(args.rnn_model_loadpath)
            rnn.load_state_dict(rnn_model_dict['state_dict'])
            optim.load_state_dict(rnn_model_dict['optimizer'])
            rnn_epoch = rnn_model_dict['epoch']
            try:
                total_passes = rnn_model_dict['total_passes']
            except:
                total_passes = total_passes
            print("loaded rnn from %s at epoch %s" %(args.rnn_model_loadpath, rnn_epoch))
        else:
            print("could not find model at %s"%args.rnn_model_loadpath)
            sys.exit()


    #test_data_name = 'episodic_vae_test_results/'
    test_data_name = 'episodic_vae_test_small/'
    #test_data_name =  'episodic_vae_test_dummy/'
    #test_data_name =  'episodic_vae_test_tiny/'
    train_data_name = test_data_name.replace('test', 'train')

    test_data_path =  os.path.join(args.datadir,test_data_name)
    train_data_path = os.path.join(args.datadir,train_data_name)

    test_data_loader = DataLoader(EpisodicFroggerDataset(test_data_path, transform=args.transform), batch_size=32, shuffle=True)
    train_data_loader = DataLoader(EpisodicFroggerDataset(train_data_path, transform=args.transform, limit=args.num_train_limit), batch_size=32, shuffle=True)
    for e in range(rnn_epoch+1,rnn_epoch+args.num_epochs):
        ep_cnt, train_losses = train(e,train_data_loader,do_use_cuda=use_cuda)
        total_passes +=ep_cnt
        test_losses = test(e,test_data_loader,do_use_cuda=use_cuda)
        train_loss_logger.log(e,np.mean(train_losses))
        test_loss_logger.log(e,np.mean(test_losses))
        print('saving epoch {} train loss {} test loss {}'.format(e,
                                                                      np.mean(train_losses),
                                                                      np.mean(test_losses)))

        if not e%args.save_every:
            state = {'epoch':e,
                    'train_loss':np.mean(train_losses),
                    'test_loss':np.mean(test_losses),
                    'state_dict':rnn.state_dict(),
                    'optimizer':optim.state_dict(),
                    'total_passes':total_passes,
                     }
            filename = os.path.join(default_base_savedir , '%s_rnn_model_epoch_%06d.pkl'%(args.savename,e))
            save_checkpoint(state, filename=filename)
            time.sleep(5)


    e+=1
    #get_pca(train_data_loader)
    embed()
