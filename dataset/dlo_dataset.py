import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from typing import final


class DLODataset(Dataset):

    def __init__(self, root_dir, train=False, num_points=2048, transform=None):
        self.root_dir = root_dir
        self.pcd_dir = []
        self.label_dir = []
        self.PCD: final(str) = "pointclouds_normed_"
        self.LABEL: final(str) = "segmentation_normed_"
        if train:
            for i in range(35):
                self.pcd_dir.extend(glob.glob(
                    os.path.join(self.root_dir, str(i).zfill(3), self.PCD + str(num_points), "*.npy")))
                self.label_dir.extend(
                    glob.glob(os.path.join(self.root_dir, str(i).zfill(3), self.LABEL + str(num_points), "*.npy")))
        else:
            for i in range(35, 40):
                self.pcd_dir.extend(glob.glob(
                    os.path.join(self.root_dir, str(i).zfill(3), self.PCD + str(num_points), "*.npy")))
                self.label_dir.extend(
                    glob.glob(os.path.join(self.root_dir, str(i).zfill(3), self.LABEL + str(num_points), "*.npy")))
        assert len(self.pcd_dir) == len(
            self.label_dir), f"length of pcd {len(self.pcd_dir)} does not match length of label {len(self.label_dir)}"

    def __len__(self):
        return len(self.pcd_dir)

    def __getitem__(self, idx):
        pcd = np.load(np.asarray(self.pcd_dir)[idx])
        label = np.load(np.asarray(self.label_dir)[idx])
        sample = {'pcd': pcd, 'label': label}
        return sample
