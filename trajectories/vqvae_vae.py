import shutil
from vae import Encoder, Decoder, VAE, latent_loss
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import FroggerDataset, VqvaeDataset, z_q_x_mean, z_q_x_std
from torchvision import datasets, transforms
from vq_vae import AutoEncoder, to_scalar
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

def test(x,model,nr_logistic_mix,do_use_cuda=False,save_img_path=None):
    x_d, z_e_x, z_q_x, latents = model(x)
    x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
    loss_1 = discretized_mix_logistic_loss(x_d,2*x-1,use_cuda=do_use_cuda)
    loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
    loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
    test_loss = to_scalar([loss_1, loss_2, loss_3])

    if save_img_path is not None:
        idx = np.random.randint(0, len(test_data))
        x_cat = torch.cat([x[idx], x_tilde[idx]], 0)
        images = x_cat.cpu().data
        oo = 0.5*np.array(x_tilde.cpu().data)[0,0]+0.5
        ii = np.array(x.cpu().data)[0,0]
        imwrite(save_img_path, oo)
        imwrite(save_img_path.replace('.png', 'orig.png'), ii)
    return test_loss

def generate_results(data_loader):
    start_time = time.time()
    for batch_idx, (data, img_names) in enumerate(data_loader):
        if use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)
        dec = vae(x)
        decr = dec.contiguous().view(dec.shape[0], 32, 10,10)
        udec = (decr*z_q_x_std)+z_q_x_mean
        # TODO - knearest neighbors
        # going to use vae.z_mean, and vae.z_std
        # look at slides from laurent and vincent - cifar summer school
        # to prune unused dimensions
        # now we have mu and sigma that we will run knn lookup within our
        # training set. then once we've mapped to the original frame that made
        # the mu/sigma - we know frame and mu and sigma
        x_d = qmodel.decoder(udec)
        x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
        nx_tilde = x_tilde.cpu().data.numpy()
        nx_tilde = (0.5*nx_tilde+0.5)*255
        nx_tilde = nx_tilde.astype(np.uint8)

        for ind, img_name in enumerate(img_names):
            print('int', ind)
            gen_img_name = img_name.replace('.npy', 'vv_gen.png')
            imwrite(gen_img_name, nx_tilde[ind][0])
        #    z_q_x_name = img_name.replace('.png', 'vqvae_z_q_x.npy')
        #    np.save(z_q_x_name, nz_q_x[ind])
        if not batch_idx%10:
            print 'Generate batch_idx: {} Time: {}'.format(
                batch_idx, time.time() - start_time
            )

if __name__ == '__main__':
    import argparse
    default_base_datadir = '../saved/'
    default_vqvae_savepath = os.path.join(default_base_datadir, 'cars_only_train.pkl')
    default_vae_savepath = os.path.join(default_base_datadir, 'vae_model.pkl')

    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-q', '--vqvae_loadpath', default=default_vqvae_savepath)
    parser.add_argument('-v', '--vae_loadpath', default=default_vae_savepath)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)

    args = parser.parse_args()
    train_data_dir = os.path.join(args.datadir, 'imgs_train')
    test_data_dir =  os.path.join(args.datadir, 'imgs_test')
    use_cuda = args.cuda

    epoch = 0
    vqvae_z = 32
    nr_logistic_mix = 10
    vae_data_dim = 10*10*32
    vae_z = 64

    if use_cuda:
        print("using gpu")
        qmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix, encoder_output_size=vqvae_z).cuda()
    else:
        qmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix, encoder_output_size=vqvae_z)

    if os.path.exists(args.vqvae_loadpath):
        qmodel_dict = torch.load(args.vqvae_loadpath)
        qmodel.load_state_dict(qmodel_dict['state_dict'])
        epoch =  qmodel_dict['epoch']
        print('loaded checkpoint at epoch: {} from {}'.format(epoch, args.vqvae_loadpath))
    else:
        print('could not find checkpoint at {}'.format(args.vqvae_loadpath))

    encoder = Encoder(vae_data_dim, vae_z)
    decoder = Decoder(vae_z, vae_data_dim)
    vae = VAE(encoder, decoder, use_cuda)
    if use_cuda:
        print("using gpu")
        vae = vae.cuda()
        vae.encoder = vae.encoder.cuda()
        vae.decoder = vae.decoder.cuda()

    if os.path.exists(args.vae_loadpath):
        vmodel_dict = torch.load(args.vae_loadpath)
        vae.load_state_dict(vmodel_dict['state_dict'])
        epoch =  vmodel_dict['epoch']
        print('loaded vae checkpoint at epoch: {} from {}'.format(epoch, args.vae_loadpath))
    else:
        print('could not find vae checkpoint at {}'.format(args.vae_loadpath))


    ##############################
    z_data_train_loader = DataLoader(VqvaeDataset(train_data_dir,
                                   transform=transforms.ToTensor(),
                                   limit=args.num_train_limit),
                                   batch_size=64, shuffle=False)

    z_data_test_loader = DataLoader(VqvaeDataset(test_data_dir,
                                   transform=transforms.ToTensor()),
                                   batch_size=64, shuffle=False)

    #generate_results(z_data_test_loader)
    generate_results(z_data_train_loader)

