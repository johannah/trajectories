import numpy as np
from IPython import embed
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os, sys
from imageio import imread

class FroggerDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        search_path = os.path.join(self.root_dir, 'seed_*.png')
        ss = sorted(glob(search_path))
        self.indexes = [s for s in ss if 'gen' not in s]

        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            raise
        if limit is not None:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]

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

        if not len(self.indexes):
            print("Error no files found at {}".format(search_path))
            raise
        if limit is not None:
            self.indexes = self.indexes[:min(len(self.indexes), limit)]

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
        # normalize v_q
        data = (data-z_q_x_mean)/z_q_x_std
        # normalize for embedding space
        #data = 2*((data/512.0)-0.5)
        return data,data_name



