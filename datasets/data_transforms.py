import numpy as np
import torch
import random


class PointcloudRotate(object):
    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)

            if self.axis == 0:
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, cosval, -sinval],
                                            [0, sinval, cosval]])
            elif self.axis == 1:
                rotation_matrix = np.array([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]])
            elif self.axis == 2:
                rotation_matrix = np.array([[cosval, -sinval, 0],
                                            [sinval, cosval, 0],
                                            [0, 0, 1]])
            else:
                raise ValueError(f"Invalid rotation axis {self.axis}. Must be 0, 1, or 2.")

            R = torch.from_numpy(rotation_matrix.astype(np.float32)).to(pc.device)
            pc[i, :, :] = torch.matmul(pc[i], R)

        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data
            
        return pc


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())
            
        return pc


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            pc[i, :, 0:3] = pc[i, :, 0:3] + torch.from_numpy(xyz2).float().cuda()
            
        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.5):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((pc.size()[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                cur_pc = pc[i, :, :]
                cur_pc[drop_idx.tolist(), 0:3] = cur_pc[0, 0:3].repeat(len(drop_idx), 1)  # set to the first point
                pc[i, :, :] = cur_pc

        return pc


class RandomHorizontalFlip(object):
  def __init__(self, upright_axis = 'z', is_temporal=False):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])


  def __call__(self, coords):
    bsize = coords.size()[0]
    for i in range(bsize):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = torch.max(coords[i, :, curr_ax])
                    coords[i, :, curr_ax] = coord_max - coords[i, :, curr_ax]
    return coords

  
class PointCloudNoise(object):
    def __init__(self, sigma=0.1, clip=0.2):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            noise = torch.randn(pc[i, :, 0:3].shape) * self.sigma
            noise = noise.clamp(-self.clip, self.clip)
            pc[i, :, 0:3] += noise.cuda()
        return pc
    
from knn_cuda import KNN
from utils import misc
import torch
import torch.nn as nn

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

def mask_groups(xyz, num_group, group_size, mask_ratio):
    group = Group(num_group, group_size)
    neighborhood, center = group(xyz)

    bsize = xyz.size()[0]
    n_points = neighborhood.size()[1] * neighborhood.size()[2]

    D = 3

    mask_points = int(n_points * mask_ratio)
    masked_neighborhood = neighborhood.clone()

    for i in range(bsize):
        for j in range(num_group):
            if random.random() < 0.5:
                continue
            group_indices = torch.arange(j * group_size, (j+1) * group_size, device=xyz.device)
            masked_indices = torch.randperm(group_size, device=xyz.device)[:mask_points]
            masked_group_indices = group_indices[masked_indices]

            masked_neighborhood[i, j, masked_indices, :] = torch.zeros(D, device=xyz.device)
            masked_center = center[i, j, :].unsqueeze(0)
            masked_centered = masked_neighborhood[i, j, :, :] - masked_center
            masked_centered[masked_indices, :] = torch.zeros(D, device=xyz.device)
            masked_neighborhood[i, j, :, :] = masked_centered + masked_center

    masked_xyz = masked_neighborhood.view(bsize, -1, D)
    return masked_xyz