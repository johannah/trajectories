import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import VqvaeDataset, z_q_x_mean, z_q_x_std, FroggerDataset
from torch import nn
from torchvision import datasets, transforms
from vae import Encoder, Decoder, VAE, latent_loss
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time
import os
from imageio import imread, imwrite
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
mse_loss = nn.MSELoss()

def train(epoch,model,optimizer,train_loader,do_checkpoint,do_use_cuda):
    latent_losses = []
    dmll_losses = []
    kl_weight = min(1.0,epoch*1e-2+.1)
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)
        optimizer.zero_grad()
        x_d = vae(x.contiguous().view(x.shape[0], -1))
        x_di = x_d.contiguous().view(x_d.shape[0], probs_size, dsize, dsize)
        dmll_loss = discretized_mix_logistic_loss(x_di, 2*x-1, nr_mix=nr_mix, use_cuda=do_use_cuda)
        #x_di = x_d.contiguous().view(x_d.shape[0], 1, dsize, dsize)
        #dmll_loss = mse_loss(x_di, x)
        kl_loss = kl_weight*latent_loss(vae.z_mean, vae.z_sigma)
        loss = dmll_loss+kl_loss
        loss.backward()
        optimizer.step()
        latent_losses.append(kl_loss.cpu().data)
        dmll_losses.append(dmll_loss.cpu().data)

        if not batch_idx%500:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tKL Loss: {} MSE Loss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / float(len(train_loader)),
                np.asarray(latent_losses).mean(0),
                np.asarray(dmll_losses).mean(0),
                time.time() - start_time
            )

    state = {'epoch':epoch,
             'state_dict':vae.state_dict(),
             'dmll_losses':np.asarray(dmll_losses).mean(0),
             'latent_losses':np.asarray(latent_losses).mean(0),
             'optimizer':optimizer.state_dict(),
             }
    return model, optimizer, state

def test(x,vae,vqvae_model,do_use_cuda=False,save_img_path=None):
    x_d = vae(x.contiguous().view(x.shape[0], -1))
    x_di = x_d.contiguous().view(x_d.shape[0], probs_size, dsize, dsize)
    xi = x.contiguous().view(x.shape[0], 1, dsize, dsize)
    dmll_loss = discretized_mix_logistic_loss(x_di, 2*xi-1, nr_mix=nr_mix, use_cuda=do_use_cuda)
    kl_loss = kl_weight*latent_loss(vae.z_mean, vae.z_sigma)
    test_loss = dmll_loss+kl_loss
    return test_loss


def generate_results(data_loader,nr_logistic_mix,do_use_cuda):
    start_time = time.time()
    for batch_idx, (data, img_names) in enumerate(data_loader):
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)
        x_d = vae(x.contiguous().view(x.shape[0], -1))
        #x_tilde = x_d.contiguous().view(x_d.shape[0], 1, dsize, dsize)
        x_di = x_d.contiguous().view(x_d.shape[0], probs_size, dsize, dsize)
        x_tilde = sample_from_discretized_mix_logistic(x_di, nr_logistic_mix)
        nx_tilde = x_tilde.cpu().data.numpy()
        inx_tilde = ((nx_tilde)*255).astype(np.uint8)
        for ind, img_name in enumerate(img_names):
            gen_img_name = img_name.replace('.png', 'pixelvae_gen.png')
            imwrite(gen_img_name, inx_tilde[ind][0])
        if not batch_idx%10:
            print 'Generate batch_idx: {} Time: {}'.format(
                batch_idx, time.time() - start_time
            )


def save_checkpoint(state, is_best=False, filename='model.pkl'):
    torch.save(state, filename)
    if is_best:
        bestpath = os.path.join(os.path.split(filename)[0], 'model_best.pkl')
        shutil.copyfile(filename, bestpath)

if __name__ == '__main__':
    import argparse
    default_base_datadir = '../saved/'
    default_model_savepath = os.path.join(default_base_datadir, 'pixel_vae_model.pkl')

    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-s', '--model_savepath', default=default_model_savepath)
    parser.add_argument('-l', '--model_loadpath', default=None)
    parser.add_argument('-e', '--num_epochs', default=150, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of imgs from pixel vae')


    args = parser.parse_args()
    train_data_dir = os.path.join(args.datadir, 'imgs_train')
    test_data_dir =  os.path.join(args.datadir, 'imgs_test')
    use_cuda = args.cuda

    dsize = 40
    data_dim = dsize*dsize
    nr_mix = 10
    # mean and scale for each components and weighting bt components (10+2*10)
    probs_size = (2*nr_mix)+nr_mix
    dout = data_dim*probs_size
    latent_size = 64

    encoder = Encoder(data_dim,  latent_size)
    decoder = Decoder(latent_size, dout)
    vae = VAE(encoder, decoder, use_cuda)
    # square error is not the correct loss - for ordered input,
    # should use softmax for unordered input ( like mine )

    if use_cuda:
        print("using gpu")
        vae = vae.cuda()
        vae.encoder = vae.encoder.cuda()
        vae.decoder = vae.decoder.cuda()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    epoch = 0
    data_train_loader = DataLoader(FroggerDataset(train_data_dir,
                                   transform=transforms.ToTensor(),
                                   limit=args.num_train_limit),
                                   batch_size=64, shuffle=True)
    data_test_loader = DataLoader(FroggerDataset(test_data_dir,
                                   transform=transforms.ToTensor()),
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

    if not args.generate_results:
        for e in xrange(epoch,epoch+args.num_epochs):
            vae, opt, state = train(e,vae,opt,data_train_loader,
                                do_checkpoint=True,do_use_cuda=use_cuda)
            #test_loss = test(x_test,vae,do_use_cuda=use_cuda,save_img_path=test_img)
            #print('test_loss {}'.format(test_loss))
            #state['test_loss'] = test_loss
            save_checkpoint(state, filename=args.model_savepath)

    else:
        generate_results(data_test_loader,nr_mix,use_cuda)
        generate_results(data_train_loader,nr_mix,use_cuda)





