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

best_inds = np.load('best_inds.npz')['arr_0']
mus = np.load('train_mu_conv_vae.npz')['arr_0']
sigmas = np.load('train_sigma_conv_vae.npz')['arr_0']
diff = np.abs(np.max(mus, axis=0) - np.min(mus, axis=0))
# filter out inds which are near zero - should be about 50 left
worst_inds = np.where(diff<1)[0]
np.savez('worst_inds.npz',worst_inds)

rdn = np.random.RandomState(3433)

def generate_reconstruction(base_path,data_loader,nr_logistic_mix,do_use_cuda):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for i, (mu,sigma) in enumerate(zip(mus,sigmas)):
        if 100 <i < 200:
            if do_use_cuda:
                tmu = Variable(torch.FloatTensor(mu), requires_grad=False).cuda()
                tsigma = Variable(torch.FloatTensor(sigma), requires_grad=False).cuda()
            else:
                tmu = Variable(torch.FloatTensor(mu), requires_grad=False)
                tsigma = Variable(torch.FloatTensor(sigma), requires_grad=False)

            bs = 20
            base = Variable(torch.from_numpy(np.zeros((bs,800))).float(), requires_grad=False)
            for s in range(bs):
                base_noise = Variable(torch.from_numpy(rdn.normal(0,1,size=tsigma.size())).float(), requires_grad=False)
                base[s] = tmu+tsigma*base_noise

            base[:,worst_inds] = 0.0
            z = base.contiguous().view(bs,32,5,5)
            x_d = vae.decoder(z)
            x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
            nx_tilde = x_tilde.cpu().data.numpy()
            inx_tilde = ((0.5*nx_tilde+0.5)*255).astype(np.uint8)
            mean_tilde = np.mean(inx_tilde, axis=0)[0].astype(np.uint8)
            max_tilde = np.max(inx_tilde, axis=0)[0].astype(np.uint8)
            mean_img_name = os.path.join(base_path, 'gmean_%05d.png'%(i))
            a_img_name = os.path.join(base_path, 'gadapt_%05d.png'%(i))
            max_img_name = os.path.join(base_path, 'gmax_%05d.png'%(i))
            imwrite(mean_img_name, mean_tilde)
            imwrite(max_img_name, max_tilde)
            nonzero = np.count_nonzero(inx_tilde,axis=0)[0]
            adapt_tilde = max_tilde
            # must have 3 instances to go into adapt
            adapt_tilde[nonzero<3] = 0
            imwrite(a_img_name, adapt_tilde)
            #for q in range(bs):
                #img_name = os.path.join(base_path, 'mu_%05d_%02d.png'%(i,q))
                #imwrite(img_name, inx_tilde[q,0])


def generate_results(base_path,data_loader,nr_logistic_mix,do_use_cuda):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
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

    generate_reconstruction(os.path.join(default_base_datadir,'train_results'), data_train_loader,nr_mix,use_cuda)
    #generate_results('../saved/test_results/', data_test_loader,nr_mix,use_cuda)
#    generate_results('../saved/train_results/', data_train_loader,nr_mix,use_cuda)








