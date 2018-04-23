import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from datasets import FroggerDataset

def train(epoch,model,optimizer,train_loader,do_checkpoint,do_use_cuda):
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)

        optimizer.zero_grad()

        x_tilde, z_e_x, z_q_x, latents = model(x)
        z_q_x.retain_grad()

        #loss_1 = F.binary_cross_entropy(x_tilde, x)
        loss_1 = discretized_mix_logistic_loss(x_tilde,2*x-1,use_cuda=do_use_cuda)
        loss_1.backward(retain_graph=True)
        model.embedding.zero_grad()
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        loss_2 = F.mse_loss(z_q_x, z_e_x.detach())
        loss_2.backward(retain_graph=True)
        loss_3 = .25*F.mse_loss(z_e_x, z_q_x.detach())
        loss_3.backward()
        optimizer.step()
        train_loss.append(to_scalar([loss_1, loss_2, loss_3]))
        if not batch_idx%10:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / float(len(train_loader)),
                np.asarray(train_loss).mean(0),
                time.time() - start_time
            )

    state = {'epoch':epoch,
             'state_dict':model.state_dict(),
             'loss':np.asarray(train_loss).mean(0),
             'optimizer':optimizer.state_dict(),
             }
    return model, optimizer, state

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

def save_checkpoint(state, is_best=False, filename='model.pkl'):
    torch.save(state, filename)
    if is_best:
        bestpath = os.path.join(os.path.split(filename)[0], 'model_best.pkl')
        shutil.copyfile(filename, bestpath)

def generate_dataset(model,data_loader,nr_logistic_mix,do_use_cuda):
    start_time = time.time()
    for batch_idx, (data, img_names) in enumerate(data_loader):
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)

        x_d, z_e_x, z_q_x, latents = model(x)
        # z_e_x is output of encoder
        # z_q_x is input into decoder
        # latents is code book
        #x_tilde = sample_from_discretized_mix_logistic(x_d, nr_logistic_mix)
        #nx_tilde = x_tilde.cpu().data.numpy()
        #nx_tilde = (0.5*nx_tilde+0.5)*255
        #nx_tilde = nx_tilde.astype(np.uint8)
        #vae_input = z_e_x.contiguous().view(z_e_x.shape[0],-1)
        #vqvae_rec_images = x_tilde.cpu().data.numpy()
        nz_q_x = z_q_x.contiguous().view(z_q_x.shape[0],-1).cpu().data.numpy()
        nlatents = latents.cpu().data.numpy()
        for ind, img_name in enumerate(img_names):
            #gen_img_name = img_name.replace('.png', 'vqvae_gen.png')
            #imwrite(gen_img_name, nx_tilde[ind][0])
            #latents_name = img_name.replace('.png', 'vqvae_latents.npy')
            z_q_x_name = img_name.replace('.png', 'vqvae_z_q_x.npy')
            np.save(z_q_x_name, nz_q_x[ind])
        if not batch_idx%10:
            print 'Generate batch_idx: {} Time: {}'.format(
                batch_idx, time.time() - start_time
            )

if __name__ == '__main__':
    import argparse
    default_base_datadir = '../saved/'
    default_model_savepath = os.path.join(default_base_datadir, 'frogger_model.pkl')

    parser = argparse.ArgumentParser(description='train vq-vae for frogger images')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument('-d', '--datadir', default=default_base_datadir)
    parser.add_argument('-s', '--model_savepath', default=default_model_savepath)
    parser.add_argument('-l', '--model_loadpath', default=None)
    parser.add_argument('-z', '--num_z', default=32, type=int)
    parser.add_argument('-e', '--num_episodes', default=150, type=int)
    parser.add_argument('-n', '--num_train_limit', default=-1, help='debug flag for limiting number of training images to use. defaults to using all images', type=int)
    parser.add_argument('-g', '--generate_dataset', action='store_true', default=False, help='generate dataset of codes')

    args = parser.parse_args()
    train_data_dir = os.path.join(args.datadir, 'imgs_train')
    test_data_dir =  os.path.join(args.datadir, 'imgs_test')
    use_cuda = args.cuda

    nr_logistic_mix = 10
    if use_cuda:
        print("using gpu")
        vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix, encoder_output_size=args.num_z).cuda()
    else:
        vmodel = AutoEncoder(nr_logistic_mix=nr_logistic_mix, encoder_output_size=args.num_z)
    opt = torch.optim.Adam(vmodel.parameters(), lr=1e-3)
    epoch = 0
    if args.model_loadpath is not None:
        if os.path.exists(args.model_loadpath):
            model_dict = torch.load(args.model_loadpath)
            vmodel.load_state_dict(model_dict['state_dict'])
            opt.load_state_dict(model_dict['optimizer'])
            epoch =  model_dict['epoch']
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, args.model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(args.model_loadpath))

    data_train_loader = DataLoader(FroggerDataset(train_data_dir,
                                   transform=transforms.ToTensor(), limit=args.num_train_limit),
                                   batch_size=64, shuffle=True)
    data_test_loader = DataLoader(FroggerDataset(test_data_dir,
                                  transform=transforms.ToTensor()),
                                  batch_size=32, shuffle=True)
    test_data = data_test_loader
    for batch_idx, (test_data, _) in enumerate(data_test_loader):
        if use_cuda:
            x_test = Variable(test_data).cuda()
        else:
            x_test = Variable(test_data)

    if not args.generate_dataset:
        test_img = args.model_savepath.replace('.pkl', '_test.png')
        for i in xrange(epoch,epoch+args.num_episodes):
            vmodel, opt, state = train(i,vmodel,opt,data_train_loader,
                                do_checkpoint=True,do_use_cuda=use_cuda)
            test_loss = test(x_test,vmodel,nr_logistic_mix,do_use_cuda=use_cuda,save_img_path=test_img)
            print('test_loss {}'.format(test_loss))
            state['test_loss'] = test_loss
            save_checkpoint(state, filename=args.model_savepath)
    else:
        generate_results(vmodel,data_test_loader,nr_logistic_mix,use_cuda)
        generate_results(vmodel,data_train_loader,nr_logistic_mix,use_cuda)






