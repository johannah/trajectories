import shutil
import torch
from IPython import embed
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

class VqvaeDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, '*.npy')
        self.indexes = glob(search_path)
        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            raise
        if limit is not None:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        data_name = self.indexes[idx]
        data = np.load(data_name).ravel().astype(np.float32)
        data = 2*((data/512.0)-0.5)
        return data,data_name


def train(epoch,model,optimizer,train_loader,do_checkpoint,do_use_cuda):
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        if do_use_cuda:
            x = Variable(data, requires_grad=False).cuda()
        else:
            x = Variable(data, requires_grad=False)
        optimizer.zero_grad()
        dec = vae(x)
        ll = latent_loss(vae.z_mean, vae.z_sigma)
        loss = criterion(dec, x)+ll
        loss.backward()
        optimizer.step()
        train_loss.append(loss.cpu().data[0])

        if not batch_idx%100:
            print 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / float(len(train_loader)),
                np.asarray(train_loss).mean(0),
                time.time() - start_time
            )

    state = {'epoch':epoch,
             'state_dict':vae.state_dict(),
             'loss':np.asarray(train_loss).mean(0),
             'optimizer':optimizer.state_dict(),
             }
    return model, optimizer, state

def test(x,vae,vqvae_model,do_use_cuda=False,save_img_path=None):
    dec = vae(x)
    ll = latent_loss(vae.z_mean, vae.z_sigma)
    loss = criterion(dec, x)+ll
    test_loss = loss.cpu().data.mean()
    return test_loss


    #if save_img_path is not None:
    #    idx = np.random.randint(0, len(test_data))
    #    x_cat = torch.cat([x[idx], x_tilde[idx]], 0)
    #    images = x_cat.cpu().data
    #    oo = 0.5*np.array(x_tilde.cpu().data)[0,0]+0.5
    #    ii = np.array(x.cpu().data)[0,0]
    #    imwrite(save_img_path, oo)
    #    imwrite(save_img_path.replace('.png', 'orig.png'), ii)
    #return test_loss

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

    data_dim = 100

    encoder = Encoder(data_dim,  args.num_z)
    decoder = Decoder(args.num_z, data_dim)
    vae = VAE(encoder, decoder)
    criterion = nn.MSELoss()
    # square error is not correct loss - for ordered input,
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
    test_data = data_train_loader

    if args.model_loadpath is not None:
        if os.path.exists(args.model_loadpath):
            model_dict = torch.load(args.model_loadpath)
            vae.load_state_dict(model_dict['state_dict'])
            opt.load_state_dict(model_dict['optimizer'])
            epoch =  model_dict['epoch']
            print('loaded checkpoint at epoch: {} from {}'.format(epoch, args.model_loadpath))
        else:
            print('could not find checkpoint at {}'.format(args.model_loadpath))

    for batch_idx, (test_data, _) in enumerate(data_test_loader):
        if use_cuda:
            x_test = Variable(test_data).cuda()
        else:
            x_test = Variable(test_data)

    test_img = args.model_savepath.replace('.pkl', '_test.png')
    for e in xrange(epoch,epoch+args.num_epochs):
        vae, opt, state = train(e,vae,opt,data_train_loader,
                            do_checkpoint=True,do_use_cuda=use_cuda)
        #test_loss = test(x_test,vae,do_use_cuda=use_cuda,save_img_path=test_img)
        #print('test_loss {}'.format(test_loss))
        #state['test_loss'] = test_loss
        save_checkpoint(state, filename=args.model_savepath)





