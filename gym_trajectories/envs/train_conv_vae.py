import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from conv_vae import Encoder, Decoder, VAE, latent_loss
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time
from glob import glob
import os
from imageio import imread, imwrite
from PIL import Image
from utils import discretized_mix_logistic_loss
from utils import sample_from_discretized_mix_logistic
from datasets import FroggerDataset

def train(epoch,model,optimizer,train_loader,do_checkpoint,do_use_cuda):
    latent_losses = []
    dmll_losses = []
    kl_weight = 1.0
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)

        optimizer.zero_grad()
        x_di = vae(x)
        dmll_loss = discretized_mix_logistic_loss(x_di, 2*x-1, nr_mix=nr_mix, use_cuda=do_use_cuda)
        kl_loss = kl_weight*latent_loss(vae.z_mean, vae.z_sigma)
        loss = dmll_loss+kl_loss
        loss.backward()
        optimizer.step()
        latent_losses.append(kl_loss.cpu().data.numpy().mean())
        dmll_losses.append(dmll_loss.cpu().data.numpy().mean())

        if not batch_idx%100:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tKL Loss: {} X Loss: {} Time: {}'.format(
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


def save_checkpoint(state, is_best=False, filename='model.pkl'):
    torch.save(state, filename)
    if is_best:
        bestpath = os.path.join(os.path.split(filename)[0], 'model_best.pkl')
        shutil.copyfile(filename, bestpath)

def generate_results(base_path,data_loader,nr_logistic_mix,do_use_cuda):
    start_time = time.time()
    data_mu = np.empty((0,800))
    data_sigma = np.empty((0,800))
    limit = 4000
    for batch_idx, (data, img_names) in enumerate(data_loader):
        if batch_idx*32 > limit:
            continue
        else:
            if do_use_cuda:
                x = Variable(data, requires_grad=False).cuda()
            else:
                x = Variable(data, requires_grad=False)

            x_d = vae(x)
            x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
            nx_tilde = x_tilde.cpu().data.numpy()
            inx_tilde = ((0.5*nx_tilde+0.5)*255).astype(np.uint8)
            # vae.z_mean is batch_sizex800
            # vae.z_sigma is batch_sizex800
            zmean = vae.z_mean.cpu().data.numpy()
            zsigma = vae.z_sigma.cpu().data.numpy()
            data_mu = np.vstack((data_mu,zmean))
            data_sigma = np.vstack((data_sigma,zsigma))
            for ind, img_path in enumerate(img_names):
                img_name = os.path.split(img_path)[1]
                gen_img_name = img_name.replace('.png', 'conv_vae_gen.png')
                #gen_latent_name = img_name.replace('.png', 'conv_vae_latents.npz')
                imwrite(os.path.join(base_path,gen_img_name), inx_tilde[ind][0])
                #np.savez(gen_latent_name, zmean=zmean[ind], zsigma=zsigma[ind])
            if not batch_idx%10:
                print 'Generate batch_idx: {} Time: {}'.format(
                    batch_idx, time.time() - start_time
                )
    np.savez(os.path.join(base_path,'mu_conv_vae.npz'), data_mu)
    np.savez(os.path.join(base_path,'sigma_conv_vae.npz'), data_sigma)

if __name__ == '__main__':
    import argparse
    default_base_datadir = '/localdata/jhansen/trajectories_frames/saved/'
    default_model_savepath = os.path.join(default_base_datadir, 'conv_vae_model.pkl')

    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-s', '--model_savepath', default=default_model_savepath)
    parser.add_argument('-l', '--model_loadpath', default=None)
    parser.add_argument('-z', '--num_z', default=32, type=int)
    parser.add_argument('-e', '--num_epochs', default=350, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_results', action='store_true', default=False, help='generate dataset of codes')

    args = parser.parse_args()
    train_data_dir = os.path.join(args.datadir, 'imgs_train')
    test_data_dir =  os.path.join(args.datadir, 'imgs_test')
    use_cuda = args.cuda

    dsize = 40
    nr_mix = 10
    # mean and scale for each components and weighting bt components (10+2*10)
    probs_size = (2*nr_mix)+nr_mix
    latent_size = 32

    encoder = Encoder(latent_size)
    decoder = Decoder(latent_size, probs_size)
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
        #generate_results('../saved/test_results/', data_test_loader,nr_mix,use_cuda)
        generate_results('../saved/train_results/', data_train_loader,nr_mix,use_cuda)








