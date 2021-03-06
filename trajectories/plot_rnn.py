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
from datasets import EpisodicFroggerDataset, EpisodicDiffFroggerDataset

from collections import OrderedDict
from imageio import imread, imwrite
from glob import glob
from vq_vae_small import AutoEncoder
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

dparams = np.load('vae_diff_params.npz')
mu_diff_mean = dparams['mu_diff_mean'][best_inds]
mu_diff_std = dparams['mu_diff_std'][best_inds]
sig_diff_mean = dparams['sig_diff_mean'][best_inds]
sig_diff_std = dparams['sig_diff_std'][best_inds]

def get_cuts(length,window_size):
    assert(window_size<length)
    st_pts = list(np.arange(0,length,window_size,dtype=np.int))
    end_pts = st_pts[1:]
    if end_pts[-1] != length:
         end_pts.append(length)
    else:
         print("cutting start")
         st_pts = st_pts[:-1]
    return zip(st_pts, end_pts)


def generate_imgs(dataloader,output_filepath,true_img_path,data_type,transform):
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    for batch_idx, (data_mu_diff_scaled, data_mu_diff, data_mu_orig, data_sigma_diff_scaled, data_sigma_diff, data_sigma_orig, name) in enumerate(dataloader):
        # data_mu_orig will be one longer than the diff versions
        batch_size = data_mu_diff_scaled.shape[0]
        # predict one less time step than availble (first is input)
        n_timesteps  = data_mu_diff_scaled.shape[1]
        vae_input_size = 800
        #######################
        # get rnn details
        #######################

        rnn_data = data_mu_diff_scaled.permute(1,0,2)
        seq = Variable(torch.FloatTensor(rnn_data), requires_grad=False)
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
        # get time offsets correct
        x = seq[:-1]
        # put initial step in
        rnn_outputs = [seq[0]]
        gt_outputs = [seq[0]]
        nrnn_outputs = [seq[0].cpu().data.numpy()]
        ngt_outputs = [seq[0].cpu().data.numpy()]
        for i in range(len(x)):
            # number of frames to start with
            #if i < 4:
            output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            #else:
            #    output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(output, h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            nrnn_outputs+=[output.cpu().data.numpy()]
            rnn_outputs+=[output]

            # put ground truth in to check pipeline
            ngt_outputs+=[seq[i+1].cpu().data.numpy()]
            gt_outputs+=[seq[i+1]]
            print(output.sum().data[0],seq[i+1].sum().data[0])

        # vae data shoud be batch,timestep(example),features
        # 0th frame is the same here
        gt_rnn_pred = torch.stack(gt_outputs, 0)
        rnn_pred = torch.stack(rnn_outputs, 0)

        # 0th frame is the same here
        rnn_mu_diff_scaled = rnn_pred.permute(1,0,2).data.numpy()
        gt_rnn_mu_diff_scaled = gt_rnn_pred.permute(1,0,2).data.numpy()

        nrnn_mu_diff_scaled =  np.swapaxes(np.array(nrnn_outputs),0,1)
        ngt_rnn_mu_diff_scaled = np.swapaxes(np.array(ngt_outputs),0,1)

        # only use relevant mus
        orig_mu_placeholder = Variable(torch.FloatTensor(np.zeros((n_timesteps, vae_input_size))), requires_grad=False)
        diff_mu_placeholder = Variable(torch.FloatTensor(np.zeros((n_timesteps, vae_input_size))), requires_grad=False)
        diff_mu_unscaled_placeholder = Variable(torch.FloatTensor(np.zeros((n_timesteps, vae_input_size))), requires_grad=False)
        diff_mu_unscaled_rnn_placeholder = Variable(torch.FloatTensor(np.zeros((n_timesteps,  vae_input_size))), requires_grad=False)
        gt_diff_mu_unscaled_rnn_placeholder = Variable(torch.FloatTensor(np.zeros((n_timesteps,  vae_input_size))), requires_grad=False)

        if transform == "std":
            print("removing standard deviation transform")
            # convert to numpy so broadcasting works
            rnn_mu_diff_unscaled = torch.FloatTensor((rnn_mu_diff_scaled*mu_diff_std)+mu_diff_mean[None])
            gt_rnn_mu_diff_unscaled = torch.FloatTensor((gt_rnn_mu_diff_scaled*mu_diff_std)+mu_diff_mean[None])
            data_mu_diff_unscaled = torch.FloatTensor((data_mu_diff_scaled.numpy()*mu_diff_std)+mu_diff_mean[None])
        else:
            print("no transform")
            rnn_mu_diff_unscaled = rnn_mu_diff_scaled
            gt_rnn_mu_diff_unscaled = gt_rnn_mu_diff_scaled
            data_mu_diff_unscaled = data_mu_diff_scaled

        # go through each distinct episode (should be length of 167)
        for e in range(batch_size):
            basename = os.path.split(name[e])[1].replace('.npz', '')
            if not e:
                print("starting %s"%basename)
            basepath = os.path.join(output_filepath, basename)
            # reconstruct rnn vae
            # now the size going through the decoder is 169x32x5x5
            # original data is one longer since there was no diff applied
            ep_mu_orig = data_mu_orig[e,1:]
            ep_mu_diff = data_mu_diff[e]
            ep_mu_diff_unscaled = data_mu_diff_unscaled[e]
            ep_mu_diff_unscaled_rnn = rnn_mu_diff_unscaled[e]
            gt_ep_mu_diff_unscaled_rnn = gt_rnn_mu_diff_unscaled[e]

            primer_frame = data_mu_orig[e,0,:]
            # need to reconstruct from original
            # get the first frame from the original dataset to add diffs to
            # data_mu_orig will be one frame longer
            # unscale the scaled version
            ep_mu_diff[0] += primer_frame
            ep_mu_diff_unscaled[0] += primer_frame
            ep_mu_diff_unscaled_rnn[0] += primer_frame
            gt_ep_mu_diff_unscaled_rnn[0] += primer_frame
            print("before diff add")
            for diff_frame in range(1,n_timesteps):
                #print("adding diff to %s" %diff_frame)
                ep_mu_diff[diff_frame] += ep_mu_diff[diff_frame-1]
                ep_mu_diff_unscaled[diff_frame] += ep_mu_diff_unscaled[diff_frame-1]
                ep_mu_diff_unscaled_rnn[diff_frame] += ep_mu_diff_unscaled_rnn[diff_frame-1]
                gt_ep_mu_diff_unscaled_rnn[diff_frame] += gt_ep_mu_diff_unscaled_rnn[diff_frame-1]

            rnn_mu_img = ep_mu_diff_unscaled_rnn.numpy()
            gt_rnn_mu_img = gt_ep_mu_diff_unscaled_rnn.numpy()
            ff,axf = plt.subplots(1,2, figsize=(5,10))
            axf[0].imshow(gt_rnn_mu_img, origin='lower')
            axf[0].set_title("gt_rnn_mu")
            axf[1].imshow(rnn_mu_img, origin='lower')
            axf[1].set_title("rnn_mu")
            ff.tight_layout()
            fimg_name = basepath+'_rnn_mu_plot.png'
            fimg_name = fimg_name.replace('_frame_%05d'%0, '')
            print("plotted %s" %fimg_name)
            plt.savefig(fimg_name)
            plt.close()


            orig_mu_placeholder[:,best_inds] = Variable(torch.FloatTensor(ep_mu_orig))
            diff_mu_placeholder[:,best_inds] = Variable(torch.FloatTensor(ep_mu_diff))
            diff_mu_unscaled_placeholder[:,best_inds] = Variable(torch.FloatTensor(ep_mu_diff_unscaled))
            diff_mu_unscaled_rnn_placeholder[:,best_inds] = Variable(torch.FloatTensor(ep_mu_diff_unscaled_rnn))
            gt_diff_mu_unscaled_rnn_placeholder[:,best_inds] = Variable(torch.FloatTensor(gt_ep_mu_diff_unscaled_rnn))
            #for i in range(1,diff_mu_unscaled_rnn_placeholder.shape[0]):
            #    diff_mu_unscaled_rnn_placeholder[i] = gt_diff_mu_unscaled_rnn_placeholder[i]
            # add a placeholder here if you want to process it
            mu_types = OrderedDict([('orig',orig_mu_placeholder),
                                 #  ('diff',diff_mu_placeholder),
                                 #  ('diff_unscaled',diff_mu_unscaled_placeholder),
                                   ('gtrnn',gt_diff_mu_unscaled_rnn_placeholder),
                                   ('rnn',diff_mu_unscaled_rnn_placeholder),
                                   ])
            mu_reconstructed = OrderedDict()
            # get reconstructed image for each type
            for xx, mu_output_name in enumerate(mu_types.keys()):
                mu_output = mu_types[mu_output_name]
                cuts = get_cuts(mu_output.shape[0], 1)
                print(mu_output_name, mu_output.sum().data[0], mu_output[0].sum().data[0])
                x_tildes = []
                for (s,e) in cuts:
                    mu_batch = mu_output[s:e]
                    # only put part of the episdoe through
                    x_d = vae.decoder(mu_batch.contiguous().view(mu_batch.shape[0], 32, 5, 5))
                    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix, deterministic=True)
                    x_tildes.append(x_tilde.cpu().data.numpy())
                nx_tilde = np.array(x_tildes)[:,0,0]
                inx_tilde = ((0.5*nx_tilde+0.5)*255).astype(np.uint8)
                mu_reconstructed[mu_output_name] = inx_tilde

            for frame_num in range(n_timesteps):
                true_img_name = os.path.join(true_img_path, basename.replace('_conv_vae', '.png')).replace('frame_%05d'%0, 'frame_%05d'%frame_num)
                true_img = imread(true_img_name)
                print("true img %s" %true_img_name)
                num_imgs = len(mu_reconstructed.keys())+1
                f, ax = plt.subplots(1,num_imgs, figsize=(3*num_imgs,3))

                ax[0].imshow(true_img, origin='lower')
                ax[0].set_title('true frame %04d'%frame_num)

                for ii, mu_output_name in enumerate(mu_reconstructed.keys()):
                    ax[ii+1].imshow(mu_reconstructed[mu_output_name][frame_num], origin='lower')
                    ax[ii+1].set_title(mu_output_name)

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

    parser.add_argument('-t', '--transform', default='std')
    parser.add_argument('-r', '--rnn_model_loadpath', default=default_rnn_model_loadpath)

    parser.add_argument('-dt', '--data_type', default='diff')
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
    test_dir = 'episodic_vae_test_tiny/'
    train_dir = test_dir.replace('test', 'train')
    gen_test_dir = test_dir.replace('episodic_', 'episodic_rnn_')
    gen_train_dir = train_dir.replace('episodic_', 'episodic_rnn_')
    test_data_path =  os.path.join(args.datadir,test_dir)
    train_data_path = os.path.join(args.datadir,train_dir)

    if args.data_type == 'diff':
        test_data_loader = DataLoader(EpisodicDiffFroggerDataset(test_data_path, transform=args.transform), batch_size=32, shuffle=True)
        #train_data_loader = DataLoader(EpisodicDiffFroggerDataset(train_data_path, transform=args.transform, limit=args.num_train_limit), shuffle=True)
    else:
        test_data_loader = DataLoader(EpisodicFroggerDataset(test_data_path, transform=args.transform), batch_size=32, shuffle=True)
        #train_data_loader = DataLoader(EpisodicFroggerDataset(train_data_path, transform=args.transform, limit=args.num_train_limit), shuffle=True)

    test_true_data_path = os.path.join(args.datadir, 'imgs_test')
    #train_true_data_path = os.path.join(args.datadir, 'imgs_train')
    generate_imgs(test_data_loader,os.path.join(args.datadir,  gen_test_dir), test_true_data_path, args.data_type, args.transform)
    #generate_imgs(train_data_loader,os.path.join(args.datadir, gen_train_dir), train_true_data_path)
    embed()



