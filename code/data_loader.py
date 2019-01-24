# load data
from __future__ import print_function, division
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class SuctionDataset(Dataset):
    """ Suction data set .. only using images """
    def __init__(self, data_file, root_path = None, joint_transform=None, transform=None, normal_transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.lines = open(data_file).readlines()
        self.joint_transform = joint_transform
        self.transform = transform
        self.normal_transform = normal_transform
        self.root_path = root_path

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        word = self.lines[idx].split(' ')
        img = Image.open(os.path.join(self.root_path, word[0])).convert('RGB')
        depth = Image.open(os.path.join(self.root_path, word[1]))
        mask = Image.open(os.path.join(self.root_path, word[2]))
        normal = Image.open(os.path.join(self.root_path, word[3][:-1]))

        if self.joint_transform is not None:
            img, mask, depth, normal = self.joint_transform(img, mask, depth, normal)

        if self.transform is not None:
            img = self.transform(img)

        if self.normal_transform is not None:
            normal = self.normal_transform(normal)

        mask = np.array(mask)
        mask = (mask[:, :, 1] == 255).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        depth = np.array(depth).astype(np.float32)
        depth -= depth.min()
        depth /= depth.max()
        depth -= 0.5
        depth = np.expand_dims(depth, axis=0)
        sample = {'img': img,
                  'mask': mask,
                  'depth': depth,
                  'normal': normal}
        return sample


class SuctionTestDataset(Dataset):
    """ Suction data set .. only using images """
    def __init__(self, data_file, root_path = None, joint_transform=None, transform=None, normal_transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.lines = open(data_file).readlines()
        self.joint_transform = joint_transform
        self.transform = transform
        self.normal_transform = normal_transform
        self.root_path = root_path

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        word = self.lines[idx].split(' ')
        img = Image.open(os.path.join(self.root_path, word[0])).convert('RGB')
        depth = Image.open(os.path.join(self.root_path, word[1]))
        normal = Image.open(os.path.join(self.root_path, word[-1][:-1]))

        if self.joint_transform is not None:
            img, depth, normal = self.joint_transform(img, depth, normal)

        if self.transform is not None:
            img = self.transform(img)

        if self.normal_transform is not None:
            normal = self.normal_transform(normal)

        depth = np.array(depth).astype(np.float32)
        depth -= depth.min()
        depth /= depth.max()
        depth -= 0.5
        depth = np.expand_dims(depth, axis=0)

        word_split = word[0].split('/')
        data_name = word_split[-1]
        sample = {'img': img,
                  'depth': depth,
                  'normal': normal,
                  'data_name': data_name}
        return sample
