import numpy as np
from scipy.spatial import cKDTree
from random import randint

def mask_pcd(pcd, mask_ratio):
    """
    This function takes a point cloud of size (1204, 3) with 1024 points of 3 coordinates in numpy.
    It receives a mask_ratio, which indicates the percentage of the point cloud that will be masked.
    It selects random points from the cloud and masks their 32 nearest neighbors, iterating until reaching the given mask_ratio.
    
    Parameters:
        - pcd: a numpy array of size (1204, 3) representing a point cloud
        - mask_ratio: a float between 0 and 1 indicating the percentage of the point cloud to be masked
        
    Returns:
        - points_masked: a numpy array of size (1204 - n_masked_points, 3) representing the original point cloud with
                         a certain proportion of masked points removed
    """
    
    # First, we calculate the number of points to be masked
    n_points = pcd.shape[0]
    n_masked_points = int(n_points * mask_ratio)
    # Create a copy of the original point cloud
    points_masked = np.copy(pcd)
    # Create a KD point search tree to search for the nearest neighbors of each point
    kdtree = cKDTree(points_masked)
    # Create an index matrix of size (n_points, 33) that will store the indices of the nearest neighbors of each point
    indices = kdtree.query(points_masked, k=33)[1]
    # Iterate until masking the desired number of points
    while n_masked_points > 0:
        idx = np.random.randint(0, n_points - 1)
        neighbor_indices = indices[idx, 1:33]
        mask = np.zeros(n_points, dtype=bool)
        mask[neighbor_indices] = True
        mask[idx] = True
        points_masked[mask] = np.nan
        n_masked_points -= 32
    
    # Use np.isnan() instead of ~np.isnan()
    mask = np.isnan(points_masked[:, 0])
    points_masked = points_masked[~mask]
    
    return points_masked


def add_noise_pcd(pcd, noise_var):
    """
    Adds noise to the point cloud data
    """
    noise = np.random.normal(0, noise_var, pcd.shape)
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

def apply_augmentation(data, augmentation_type, npoints, kwargs):
    return resample_pcd(augmentation_type(data, **kwargs), npoints)

def siamese_augmentation(data, config):
    if config.augmentations.type == "masks":
        data1 = apply_augmentation(data, mask_pcd, config.npoints,
                                   {"mask_ratio": config.augmentations.noise_var})
        data2 = apply_augmentation(data, mask_pcd, config.npoints,
                                      {"mask_ratio": config.augmentations.noise_var})
    elif config.augmentations.type == "noise":
        data1 = apply_augmentation(data, add_noise_pcd, config.npoints,
                                      {"noise_var": config.augmentations.noise_var})
        data2 = apply_augmentation(data, add_noise_pcd, config.npoints,
                                        {"noise_var": config.augmentations.noise_var})
    if config.augmentations.type == "random":
        data1 = apply_augmentation(data, add_noise_pcd, config.npoints,
                                    {"noise_var": config.augmentations.noise_var})
        data1 = apply_augmentation(data, mask_pcd, config.npoints,
                                    {"mask_ratio": config.augmentations.mask_ratio})
        data2 = apply_augmentation(data, add_noise_pcd, config.npoints,
                                    {"noise_var": config.augmentations.noise_var}) 
        data2 = apply_augmentation(data, mask_pcd, config.npoints,
                                        {"mask_ratio": config.augmentations.mask_ratio})
    return data1, data2
