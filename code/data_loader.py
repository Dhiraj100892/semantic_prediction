# load data
from __future__ import print_function, division
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class SuctionDataset(Dataset):
    """ Suction data set .. only using images """
    def __init__(self, data_file, root_path = None, joint_transform=None, transform=None):
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
        mask = Image.open(os.path.join(self.root_path, word[1][:-1]))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.transform is not None:
            img = self.transform(img)

        mask = np.array(mask)
        mask = (mask == 255).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        sample = {'img': img,
                  'mask': mask}
        return sample


class SuctionTestDataset(Dataset):
    """ Suction data set .. only using images """
    def __init__(self, data_file, root_path = None, joint_transform=None, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.lines = open(data_file).readlines()
        self.joint_transform = joint_transform
        self.transform = transform
        self.root_path = root_path

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        word = self.lines[idx][:-1]
        img = Image.open(os.path.join(self.root_path, word)).convert('RGB')

        if self.joint_transform is not None:
            img = self.joint_transform(img)

        if self.transform is not None:
            img = self.transform(img)

        word_split = word.split('/')
        data_name = word_split[-1]
        sample = {'img': img,
                  'data_name': data_name}
        return sample
