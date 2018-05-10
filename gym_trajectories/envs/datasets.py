import torch
import numpy as np
from IPython import embed
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os, sys
from imageio import imread

pcad = np.load('pca_components_vae.npz')
V = pcad['V']
vae_mu_mean = pcad['Xmean']
vae_mu_std = pcad['Xstd']
Xpca_std = pcad['Xpca_std']

worst_inds = np.load('worst_inds.npz')['arr_0']
all_inds = range(800)
best_inds = np.array([w for w in all_inds if w not in list(worst_inds)])

class FroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, 'seed_*.png')
        ss = sorted(glob(search_path))
        self.indexes = [s for s in ss if 'gen' not in s]
        print("found %s files in %s" %(len(self.indexes), search_path))

        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            sys.exit()
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_name = self.indexes[idx]
        image = imread(img_name)
        image = image[:,:,None].astype(np.float32)
        if self.transform is not None:
            image = self.transform(image)

        return image,img_name

class FlattenedFroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, 'seed_*.png')
        ss = sorted(glob(search_path))
        self.indexes = [s for s in ss if 'gen' not in s]
        print("found %s files in %s" %(len(self.indexes), search_path))

        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            raise
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        img_name = self.indexes[idx]
        image = imread(img_name)
        image = image[:,:,None].astype(np.float32)
        # TODO - this is non-normalized
        image = np.array(image.ravel())
        return image,img_name


z_q_x_mean = 0.16138
z_q_x_std = 0.7934

class VqvaeDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, '*z_q_x.npy')
        self.indexes = sorted(glob(search_path))
        print("found %s files in %s" %(len(self.indexes), search_path))
        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            raise
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        data_name = self.indexes[idx]
        data = np.load(data_name).ravel().astype(np.float32)
        # normalize v_q
        data = (data-z_q_x_mean)/z_q_x_std
        # normalize for embedding space
        #data = 2*((data/512.0)-0.5)
        return data,data_name

class EpisodicFroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=-1, search='*conv_vae.npz'):
        # what really matters is the seed - only generated one game per seed
        #seed_00334_episode_00029_frame_00162.png
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, search)
        self.indexes = sorted(glob(search_path))
        print("will use transform:%s"%transform)
        print("found %s files in %s" %(len(self.indexes), search_path))
        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            sys.exit()
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        dname = self.indexes[idx]
        d = np.load(open(dname, 'rb'))
        mu = d['mu'].astype(np.float32)[:,best_inds]
        sig = d['sigma'].astype(np.float32)[:,best_inds]
        if self.transform == 'pca':
            if not idx:
                print("tranforming dataset using pca")
            mu_scaled = mu-vae_mu_mean
            mu_scaled = (np.dot(mu_scaled, V.T)/Xpca_std).astype(np.float32)
        elif self.transform == 'std':
            mu_scaled= ((mu-vae_mu_mean)/vae_mu_std).astype(np.float32)
        else:
            mu_scaled = mu
            sig_scaled = sig

        return mu_scaled,mu,sig_scaled,sig,dname


class EpisodicDiffFroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=-1, search='*conv_vae.npz'):
        # what really matters is the seed - only generated one game per seed
        #seed_00334_episode_00029_frame_00162.png
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, search)
        self.indexes = sorted(glob(search_path))
        dparams = np.load('vae_diff_params.npz')
        self.mu_diff_mean = dparams['mu_diff_mean'][best_inds]
        self.mu_diff_std = dparams['mu_diff_std'][best_inds]
        self.sig_diff_mean = dparams['sig_diff_mean'][best_inds]
        self.sig_diff_std = dparams['sig_diff_std'][best_inds]
        print("will use transform:%s"%transform)
        print("found %s files in %s" %(len(self.indexes), search_path))
        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            sys.exit()
        if limit > 0:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]
            print('limited to first %s examples' %len(self.indexes))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        if idx == 0:
            print("loading first file")
        dname = self.indexes[idx]
        d = np.load(open(dname, 'rb'))
        mu = d['mu'].astype(np.float32)[:,best_inds]
        sig = d['sigma'].astype(np.float32)[:,best_inds]
        mu_diff = np.diff(mu,n=1,axis=0)
        sig_diff = np.diff(sig,n=1,axis=0)
        if self.transform == 'std':
            mu_scaled= ((mu_diff-self.mu_diff_mean)/self.mu_diff_std).astype(np.float32)
            sig_scaled= ((sig_diff-self.sig_diff_mean)/self.sig_diff_std).astype(np.float32)
        else:
            mu_scaled = mu_diff
            sig_scaled = sig_diff
        return mu_scaled,mu_diff,sig_scaled,sig_diff,dname




