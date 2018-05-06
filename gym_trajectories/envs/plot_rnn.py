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

#def train(e,dataloader,do_save=False,do_use_cuda=False):
#    losses = []
#    for batch_idx, (data_mu, data_sigma, name) in enumerate(dataloader):
#        optim.zero_grad()
#        batch_size = data_mu.shape[0]
#        # only use relevant mus
#        # data shoud be timestep,batchsize,features
#        data = data_mu[:,:,best_inds].permute(1,0,2)
#        if do_use_cuda:
#            x = Variable(torch.FloatTensor(data), requires_grad=False).cuda()
#            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
#            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
#            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
#            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
#        else:
#            x = Variable(torch.FloatTensor(data), requires_grad=False)
#            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
#            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
#            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
#            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
#        outputs = []
#        for i in range(len(x)):
#            output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
#            outputs+=[output]
#        pred = torch.stack(outputs, 0)
#        mse_loss = ((pred-x)**2).mean()
#        mse_loss.backward()
#        clip = 10
#        for p in rnn.parameters():
#            p.grad.data.clamp_(min=-clip,max=clip)
#        optim.step()
#        ll = mse_loss.cpu().data.numpy()[0]
#        if np.isnan(ll):
#            embed()
#        losses.append(ll)
#        if not batch_idx%10:
#            print('epoch {} batch_idx {} loss {}'.format(e,batch_idx,ll))
#    if do_save:
#        print('saving epoch {} loss {}'.format(e,np.mean(losses)))
#        state = {'epoch':e,
#                'loss':np.mean(losses),
#                'state_dict':rnn.state_dict(),
#                'optimizer':optim.state_dict(),
#                 }
#        filename = os.path.join(default_base_savedir , 'rnn_model_epoch_%06d_loss%05f.pkl'%(e,np.mean(losses)))
#        save_checkpoint(state, filename=filename)
#        time.sleep(5)

#def save_checkpoint(state, filename='model.pkl'):
#    print("starting save of {}".format(filename))
#    f = open(filename, 'w')
#    torch.save(state, f)
#    f.close()
#    print("finishing save of {}".format(filename))


def generate_imgs(dataloader,output_filepath,true_img_path):
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    for batch_idx, (data_mu, data_sigma, name) in enumerate(dataloader):
        batch_size = data_mu.shape[0]
        # only use relevant mus
        # rnn input data shoud be timestep,batchsize,features
        data = data_mu[:,:,best_inds].permute(1,0,2)
        if use_cuda:
            mus_vae = Variable(data_mu, requires_grad=False).cuda()
            seq = Variable(torch.FloatTensor(data), requires_grad=False).cuda()
            out_mu = Variable(torch.FloatTensor(np.zeros_like(data_mu)), requires_grad=False).cuda()
            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False).cuda()
        else:
            mus_vae = Variable(data_mu, requires_grad=False)
            seq = Variable(torch.FloatTensor(data), requires_grad=False)
            out_mu = Variable(torch.FloatTensor(np.zeros_like(data_mu)), requires_grad=False)
            h1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            c1_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            h2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
            c2_tm1 = Variable(torch.FloatTensor(np.zeros((batch_size, hidden_size))), requires_grad=False)
        outputs = []
        y = seq[1:]
        x = seq[:-1]
        for i in range(len(x)):
            output, h1_tm1, c1_tm1, h2_tm1, c2_tm1 = rnn(x[i], h1_tm1, c1_tm1, h2_tm1, c2_tm1)
            outputs+=[output]
        pred = torch.stack(outputs, 0)

        # vae data shoud be batch,timestep(example),features
        out_mu = out_mu[:,1:,:]
        out_mu[:,:,best_inds] = pred.permute(0,1,2)
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

    parser.add_argument('-r', '--rnn_model_loadpath', default=default_rnn_model_loadpath)

    parser.add_argument('-z', '--num_z', default=32, type=int)
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
    hidden_size = 512
    # input after only good parts of vae taken
    input_size = 50
    seq_length = 169
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


    test_dir = 'episodic_vae_test_results'
    #test_dir = 'episodic_vae_train_tiny/
    train_dir = test_dir.replace('test', 'train')
    gen_test_dir = test_dir.replace('episodic_', 'episodic_rnn_')
    gen_train_dir = train_dir.replace('episodic_', 'episodic_rnn_')
    test_data_path =  os.path.join(args.datadir,test_dir)
    train_data_path = os.path.join(args.datadir,train_dir)

    test_data_loader = DataLoader(EpisodicFroggerDataset(test_data_path), batch_size=32, shuffle=False)
    train_data_loader = DataLoader(EpisodicFroggerDataset(train_data_path, limit=args.num_train_limit), batch_size=32, shuffle=False)
    test_true_data_path = os.path.join(args.datadir, 'imgs_test')
    train_true_data_path = os.path.join(args.datadir, 'imgs_train')
    generate_imgs(test_data_loader,os.path.join(args.datadir,  gen_test_dir), test_true_data_path)
    generate_imgs(train_data_loader,os.path.join(args.datadir, gen_train_dir), train_true_data_path)
    embed()



