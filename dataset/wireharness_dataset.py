import torch
import numpy as np
import os
import glob
from dataset.utils import *
from torch.utils.data import Dataset
from typing import final


class WireHarenessDataset(Dataset):

    def __init__(self, root_dir: str, split="train", num_points=2048, transform=False):
        self.root_dir: str = root_dir
        self.pcd_dir: list = []
        self.label_dir: list = []
        self.transform: bool = transform
        self.PCD: final(str) = "pointclouds_normed_"
        self.LABEL: final(str) = "segmentation_normed_"
        self.split = split
        if self.split in "train":
            for i in range(32):
                self.pcd_dir.extend(glob.glob(
                    os.path.join(self.root_dir, str(i).zfill(3), self.PCD + str(num_points), "*.npy")))
                self.label_dir.extend(
                    glob.glob(os.path.join(self.root_dir, str(i).zfill(3), self.LABEL + str(num_points), "*.npy")))
        elif self.split in "val":
            for i in range(32, 36):
                self.pcd_dir.extend(glob.glob(
                    os.path.join(self.root_dir, str(i).zfill(3), self.PCD + str(num_points), "*.npy")))
                self.label_dir.extend(
                    glob.glob(os.path.join(self.root_dir, str(i).zfill(3), self.LABEL + str(num_points), "*.npy")))
        else:
            for i in range(36, 40):
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
        label[label == 0] = 0
        label[label == 1] = 0
        label[label == 2] = 1
        label[label == 3] = 0
        label[label == 4] = 2
        if self.transform and self.split == 'train':
            pcd = rotate_point_cloud_z(pcd)
            pcd = rotate_perturbation_point_cloud(pcd, angle_sigma=0.06, angle_clip=0.18)
            pcd = jitter_point_cloud(pcd, clip=0.01)
            pcd = shift_point_cloud(pcd, shift_range=0.1)
            pcd = random_scale_point_cloud(pcd, scale_low=0.5, scale_high=1.25)
            pcd, label, _ = shuffle_points(pcd, label)
        sample = {'pcd': pcd, 'label': label}
        return sample
