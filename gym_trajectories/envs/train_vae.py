import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import VqvaeDataset, z_q_x_mean, z_q_x_std
from torch import nn
from torchvision import datasets, transforms
from vae import Encoder, Decoder, VAE, latent_loss
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time
from glob import glob
import os
from imageio import imread, imwrite
from PIL import Image

def train(epoch,model,optimizer,train_loader,do_checkpoint,do_use_cuda):
    latent_losses = []
    mse_losses = []
    kl_weight = min(1.0,epoch*1e-2)
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)
        optimizer.zero_grad()
        dec = vae(x)
        kl = kl_weight*latent_loss(vae.z_mean, vae.z_sigma)
        mse_loss = criterion(dec, x)
        loss = mse_loss+kl
        loss.backward()
        optimizer.step()
        latent_losses.append(kl.cpu().data)
        mse_losses.append(mse_loss.cpu().data)

        if not batch_idx%500:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tKL Loss: {} MSE Loss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / float(len(train_loader)),
                np.asarray(latent_losses).mean(0),
                np.asarray(mse_losses).mean(0),
                time.time() - start_time
            )

    state = {'epoch':epoch,
             'state_dict':vae.state_dict(),
             'mse_losses':np.asarray(mse_losses).mean(0),
             'latent_losses':np.asarray(latent_losses).mean(0),
             'optimizer':optimizer.state_dict(),
             }
    return model, optimizer, state

def test(x,vae,vqvae_model,do_use_cuda=False,save_img_path=None):
    dec = vae(x)
    kl = latent_loss(vae.z_mean, vae.z_sigma)
    loss = criterion(dec, x)+kl
    test_loss = loss.cpu().data.mean()
    return test_loss


def save_checkpoint(state, is_best=False, filename='model.pkl'):
    torch.save(state, filename)
    if is_best:
        bestpath = os.path.join(os.path.split(filename)[0], 'model_best.pkl')
        shutil.copyfile(filename, bestpath)

if __name__ == '__main__':
    import argparse
    default_base_datadir = '../saved/'
    default_model_savepath = os.path.join(default_base_datadir, 'vae_model.pkl')

    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-s', '--model_savepath', default=default_model_savepath)
    parser.add_argument('-l', '--model_loadpath', default=None)
    parser.add_argument('-z', '--num_z', default=64, type=int)
    parser.add_argument('-e', '--num_epochs', default=150, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)

    args = parser.parse_args()
    train_data_dir = os.path.join(args.datadir, 'imgs_train')
    test_data_dir =  os.path.join(args.datadir, 'imgs_test')
    use_cuda = args.cuda

    data_dim = 10*10*32
    encoder = Encoder(data_dim,  args.num_z)
    decoder = Decoder(args.num_z, data_dim)
    vae = VAE(encoder, decoder, use_cuda)
    criterion = nn.MSELoss()
    # square error is not the correct loss - for ordered input,
    # should use softmax for unordered input ( like mine )

    if use_cuda:
        print("using gpu")
        vae = vae.cuda()
        vae.encoder = vae.encoder.cuda()
        vae.decoder = vae.decoder.cuda()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    epoch = 0
    data_train_loader = DataLoader(VqvaeDataset(train_data_dir,
                                   limit=args.num_train_limit),
                                   batch_size=64, shuffle=True)
    data_test_loader = DataLoader(VqvaeDataset(test_data_dir),
                                  batch_size=32, shuffle=True)
    test_data = data_test_loader

    if args.model_loadpath is not None:
        if os.path.exists(args.model_loadpath):
            model_dict = torch.load(args.model_loadpath)
            vae.load_state_dict(model_dict['state_dict'])
            opt.load_state_dict(model_dict['optimizer'])
            epoch =  model_dict['epoch']
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, args.model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(args.model_loadpath))
            embed()

    for batch_idx, (test_data, _) in enumerate(data_test_loader):
        if use_cuda:
            x_test = Variable(test_data).cuda()
        else:
            x_test = Variable(test_data)

    for e in xrange(epoch,epoch+args.num_epochs):
        vae, opt, state = train(e,vae,opt,data_train_loader,
                            do_checkpoint=True,do_use_cuda=use_cuda)
        #test_loss = test(x_test,vae,do_use_cuda=use_cuda,save_img_path=test_img)
        #print('test_loss {}'.format(test_loss))
        #state['test_loss'] = test_loss
        save_checkpoint(state, filename=args.model_savepath)





