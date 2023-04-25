import numpy as np
from random import randint


def mask_pcd(pcd, num_masks, augmentation_size):
    """
    Makes masks in the point cloud data
    Parameters
    ----------
    pcd : float[n,3]
        point cloud data of n size in x, y, z format
    num_masks : int
        number of holes to make
    augmentation_size : float
        size of the holes normalized, by default 0.1
    Returns
    -------
    float[n,3]
        point cloud data in x, y, z of n size format
    """
    new_pcd = pcd.copy()
    for i in range(num_masks):
        rand_point = new_pcd[randint(0, new_pcd.shape[0]-1)]  # Pick a random mask center
        dists = np.linalg.norm(new_pcd - rand_point, axis=1)  # Compute distance from each point to the hole center
        hole_mask = dists < augmentation_size  # Create a boolean mask indicating which points are inside the hole
        new_pcd = new_pcd[~hole_mask]  # Remove the points inside the hole
    return new_pcd

def add_noise_pcd(pcd, augmentation_size):
    """
    Adds noise to the point cloud data
    """
    noise = np.random.normal(0, augmentation_size, pcd.shape)
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

def siamese_augmentation(data, config):
    if config.augmentations.type == "masks":
        data1 = resample_pcd(
            mask_pcd(
                data,
                num_masks=config.augmentations.num_masks,
                augmentation_size=config.augmentations.augmentation_size,
            ),
            config.npoints,
        )
        data2 = resample_pcd(
            mask_pcd(
                data,
                num_masks=config.augmentations.num_masks,
                augmentation_size=config.augmentations.augmentation_size,
            ),
            config.npoints,
        )
    elif config.augmentations.type == "noise":
        data1 = resample_pcd(
            add_noise_pcd(data, augmentation_size=config.augmentations.augmentation_size),
            config.npoints,
        )
        data2 = resample_pcd(
            add_noise_pcd(data, augmentation_size=config.augmentations.augmentation_size),
            config.npoints,
        )
    return data1, data2
