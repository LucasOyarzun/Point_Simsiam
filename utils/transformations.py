import numpy as np
import torch
from numpy.random import randint

def make_holes_pcd(pcd, num_holes, t_size=0.1):
    """
    Makes holes in the point cloud data
    Parameters
    ----------
    pcd : np.array [N, 3]
        Point cloud data
    """
    new_pcd = pcd
    for i in range(num_holes):
        rand_point = new_pcd[
            randint(0, new_pcd.shape[0])
        ]  # Pick a random hole center
        partial_pcd = []
        for i in range(new_pcd.shape[0]):  # Check if the point is in the hole
            dist = np.linalg.norm(rand_point - new_pcd[i])
            if (
                dist >= t_size
            ):  # If not, add it to the partial point cloud
                partial_pcd = partial_pcd + [new_pcd[i]]
        new_pcd = np.array([np.array(e) for e in partial_pcd])
    return new_pcd

def add_noise_pcd(pcd, t_size=0.1):
    """
    Adds noise to the point cloud data
    """
    noise = np.random.normal(0, t_size, pcd.shape)
    return pcd + noise

def resample_pcd(pcd, n):
    """
    Drop or duplicate points so that pcd has exactly n points
    """
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])]
        )
    return pcd[idx[:n]]

def get_transformations(data, npoints, config):
    """
    Receives a batch with pcs and applies the transformations
    """
    # Permute to have the batch at the end
    #data = data.permute(1, 2, 0)
    t_type = config.type
    t_kwargs = config.kwargs
    if t_type == "holes":
        data1 = torch.stack([torch.from_numpy(resample_pcd(make_holes_pcd(pcd.numpy(), **t_kwargs), npoints)).float() for pcd in data])
        data2 = torch.stack([torch.from_numpy(resample_pcd(make_holes_pcd(pcd.numpy(), **t_kwargs), npoints)).float() for pcd in data])
    elif t_type == "noise":
        data1 = torch.stack([torch.from_numpy(add_noise_pcd(pcd.numpy(), **t_kwargs)).float() for pcd in data])
        data2 = torch.stack([torch.from_numpy(add_noise_pcd(pcd.numpy(), **t_kwargs)).float() for pcd in data])
    return data1, data2
