import os
import torch
import numpy as np
import torch.utils.data as data
import glob
from .io import IO
from .build import DATASETS
from utils.logger import *


@DATASETS.register_module()
class SimpleShape(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS

        self.sample_points_num = config.npoints
        file_list = glob.glob(os.path.join(self.data_root, "*.npz"))

        self.file_list = []

        for fi in file_list:
            model_id = fi.split("/")[-1].split("s")[1].split(".")[0]
            self.file_list.append(
                {"taxonomy_id": 0, "model_id": model_id, "file_path": fi}
            )

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def resample_pcd(self, pcd, n):
        """Drop or duplicate points so that pcd has exactly n points"""
        idx = np.random.permutation(pcd.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate(
                [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])]
            )
        return pcd[idx[:n]]

    def random_sample(self, pc, num):
        pc = self.resample_pcd(pc, self.npoints)
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(sample["file_path"]).astype(np.float32)

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample["taxonomy_id"], sample["model_id"], data

    def __len__(self):
        return len(self.file_list)
