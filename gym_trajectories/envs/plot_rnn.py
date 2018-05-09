# from KK
import matplotlib
matplotlib.use('Agg')
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

from imageio import imread, imwrite
from glob import glob
from vq_vae_small import AutoEncoder, to_scalar
from conv_vae import Encoder, Decoder, VAE
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
worst_inds = np.load('worst_inds.npz')['arr_0']
all_inds = range(800)
best_inds = np.array([w for w in all_inds if w not in list(worst_inds)])

torch.manual_seed(139)

pcad = np.load('pca_components_vae.npz')
V = pcad['V']
vae_mu_mean = pcad['Xmean']
vae_mu_std = pcad['Xstd']
Xpca_std = pcad['Xpca_std']

def generate_imgs(dataloader,output_filepath,true_img_path):
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    for batch_idx, (data_mu_scaled, data_mu_orig, data_sigma, name) in enumerate(dataloader):
        batch_size = data_mu_scaled.shape[0]
        n_timesteps  = data_mu_scaled.shape[1]-1
        # only use relevant mus
        # rnn input data shoud be timestep,batchsize,features
        data = data_mu_scaled.permute(1,0,2)
        seq = Variable(torch.FloatTensor(data), requires_grad=False)
        out_mu = Variable(torch.FloatTensor(np.zeros((batch_size,  n_timesteps, 800))), requires_grad=False)
        mus_vae = Variable(torch.FloatTensor(np.zeros((batch_size, n_timesteps, 800))), requires_grad=False)
        h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
        c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
        h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
        c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
        if use_cuda:
            mus_vae = mus_vae.cuda()
            seq = seq.cuda()
            out_mu = out_mu.cuda()
            h1_tm1 = h1_tm1.cuda()
            c1_tm1 = c1_tm1.cuda()
            h2_tm1 = h2_tm1.cuda()
            c2_tm1 = c2_tm1.cuda()
        outputs = []
        # get time offsets correct
        y = seq[1:]
        x = seq[:-1]
        data_mu_scaled = data_mu_scaled[:,1:,:]
        data_mu_orig = data_mu_orig[:,1:,:]
        for i in range(len(x)):
            # number of frames to start with
            if i < 4:
                output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            else:
                output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(output, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            outputs+=[output]
        pred = torch.stack(outputs, 0)

        # vae data shoud be batch,timestep(example),features
        pred_p = pred.permute(1,0,2).data.numpy()
        data_mu_scaled = data_mu_scaled.numpy()
        if args.transform == 'pca':
            # unnormalize data
            # how it was scaled -
            #mu_scaled = (np.dot((mu-vae_mu_mean, V.T)/Xpca_std).astype(np.float32)
            out_mu_unscaled = pred_p*Xpca_std[None,None]
            out_mu_unscaled = np.dot(out_mu_unscaled, V)+vae_mu_mean[None,None]

            vae_mu = data_mu_scaled*Xpca_std[None,None]
            vae_mu_unscaled = np.dot(vae_mu, V)+vae_mu_mean[None,None]

        elif args.transform == 'std':
            out_mu_unscaled = (pred_p*vae_mu_std)+vae_mu_mean[None,None]
            # good
            vae_mu_unscaled = (data_mu_scaled*vae_mu_std)+vae_mu_mean[None,None]
        else:
            # NO SCALING ON INPUT (NOT LIKELY)
            out_mu_unscaled = pred_p
            vae_mu_unscaled = data_mu_scaled

        out_mu[:,:,best_inds] = Variable(torch.FloatTensor(out_mu_unscaled))
        mus_vae[:,:,best_inds] = Variable(torch.FloatTensor(vae_mu_unscaled))
        # go through each distinct episode:
        for e in range(out_mu.shape[0]):
            basename = os.path.split(name[e])[1].replace('.npz', '')
            if not e:
                print("starting %s"%basename)
            basepath = os.path.join(output_filepath, basename)
            # reconstruct rnn vae
            # now the size going through the decoder is 169x32x5x5
            ep_mus = out_mu[e]
            x_d = vae.decoder(ep_mus.contiguous().view(ep_mus.shape[0], 32, 5, 5))

            x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
            nx_tilde = x_tilde.cpu().data.numpy()
            inx_tilde = ((0.5*nx_tilde+0.5)*255).astype(np.uint8)

            ep_mus_vae = mus_vae[e]
            # only use vae reconstruction
            x_d_vae = vae.decoder(ep_mus_vae.contiguous().view(ep_mus_vae.shape[0], 32, 5, 5))
            x_tilde_vae = sample_from_discretized_mix_logistic(x_d_vae, nr_logistic_mix)
            nx_tilde_vae = x_tilde_vae.cpu().data.numpy()
            inx_tilde_vae = ((0.5*nx_tilde_vae+0.5)*255).astype(np.uint8)

            for frame in range(inx_tilde.shape[0]):
                frame_num = frame+1 # pred one timestep ahead, assume 0 is given
                true_img_name = os.path.join(true_img_path, basename.replace('_conv_vae', '.png'))
                true_img_name = true_img_name.replace('frame_%05d'%0, 'frame_%05d'%frame_num)
                true_img = imread(true_img_name)
                print("true img %s" %true_img_name)
                f, ax = plt.subplots(1,3, figsize=(6,3))
                ax[0].imshow(true_img, origin='lower')
                ax[0].set_title('true frame %04d'%frame_num)
                ax[1].imshow(inx_tilde_vae[frame][0], origin='lower')
                ax[1].set_title('vae decoder')
                ax[2].imshow(inx_tilde[frame][0], origin='lower')
                ax[2].set_title('rnn vae decoder')
                f.tight_layout()
                img_name = basepath+'_rnn_plot.png'
                img_name = img_name.replace('frame_%05d'%0, 'frame_%05d'%frame_num)
                print("plotted %s" %img_name)
                plt.savefig(img_name)
                plt.close()


if __name__ == '__main__':
    import argparse
    default_base_datadir = '/localdata/jhansen/trajectories_frames/dataset/'
    default_base_savedir = '/localdata/jhansen/trajectories_frames/saved/'
    default_vae_model_loadpath = os.path.join(default_base_savedir, 'conv_vae.pkl')
    #default_rnn_model_loadpath = os.path.join(default_base_savedir, 'rnn_vae.pkl')
    default_rnn_model_loadpath = os.path.join(default_base_savedir, 'rnn_model_epoch_000152_loss0.000166.pkl')
    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-v', '--vae_model_loadpath', default=default_vae_model_loadpath)

    parser.add_argument('-t', '--transform', default='None')
    parser.add_argument('-r', '--rnn_model_loadpath', default=default_rnn_model_loadpath)

    parser.add_argument('-hs', '--hidden_size', default=512, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')

    args = parser.parse_args()
    use_cuda = args.cuda

    dsize = 40
    nr_mix = nr_logistic_mix = 10
    ## mean and scale for each components and weighting bt components (10+2*10)
    probs_size = (2*nr_mix)+nr_mix
    latent_size = 32

    encoder = Encoder(latent_size)
    decoder = Decoder(latent_size, probs_size)
    vae = VAE(encoder, decoder, use_cuda)
    if use_cuda:
        print("using gpu")
        vae = vae.cuda()
        vae.encoder = vae.encoder.cuda()
        vae.decoder = vae.decoder.cuda()
    vae_epoch = 0
    if args.vae_model_loadpath is not None:
        if os.path.exists(args.vae_model_loadpath):
            vae_model_dict = torch.load(args.vae_model_loadpath)
            vae.load_state_dict(vae_model_dict['state_dict'])
            vae_epoch =  vae_model_dict['epoch']
            print('loaded vae checkpoint at epoch: {} from {}'.format(vae_epoch, args.vae_model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(args.vae_model_loadpath))
            embed()
    else:
        print("no VAE path provided")
    # setup rnn
    hidden_size = args.hidden_size
    # input after only good parts of vae taken
    input_size = 50
    seq_length = 168
    lr = 1e-4
    rnn = RNN(input_size,hidden_size)
    optim = optim.Adam(rnn.parameters(), lr=lr, weight_decay=1e-6)
    if use_cuda:
        rnn.cuda()
    rnn_epoch = 0

    if args.rnn_model_loadpath is not None:
        if os.path.exists(args.rnn_model_loadpath):
            rnn_model_dict = torch.load(args.rnn_model_loadpath)
            rnn.load_state_dict(rnn_model_dict['state_dict'])
            rnn_epoch = rnn_model_dict['epoch']
            print('loaded rnn checkpoint at epoch: {} from {}'.format(rnn_epoch, args.rnn_model_loadpath))
        else:
            print('could not find rnn checkpoint at {}'.format(args.rnn_model_loadpath))
            embed()
    else:
        print("no RNN path provided")


    #test_dir = 'episodic_vae_test_results'
    #test_dir = 'episodic_vae_test_tiny/'
    test_dir = 'episodic_vae_test_small/'
    train_dir = test_dir.replace('test', 'train')
    gen_test_dir = test_dir.replace('episodic_', 'episodic_rnn_')
    gen_train_dir = train_dir.replace('episodic_', 'episodic_rnn_')
    test_data_path =  os.path.join(args.datadir,test_dir)
    train_data_path = os.path.join(args.datadir,train_dir)

    test_data_loader = DataLoader(EpisodicFroggerDataset(test_data_path, transform=args.transform), batch_size=32, shuffle=False)
    train_data_loader = DataLoader(EpisodicFroggerDataset(train_data_path, limit=args.num_train_limit, transform=args.transform), batch_size=32, shuffle=False)
    test_true_data_path = os.path.join(args.datadir, 'imgs_test')
    train_true_data_path = os.path.join(args.datadir, 'imgs_train')
    generate_imgs(test_data_loader,os.path.join(args.datadir,  gen_test_dir), test_true_data_path)
    #generate_imgs(train_data_loader,os.path.join(args.datadir, gen_train_dir), train_true_data_path)
    embed()



