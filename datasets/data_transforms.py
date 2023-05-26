import numpy as np
import torch
import torch.nn as nn
import random
from knn_cuda import KNN
from utils import misc



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

class PointCloudMask(object):
    def __init__(self, num_group=32, group_size=32, mask_ratio=[0.5, 0.5]):
        self.num_group = num_group
        self.group_size = group_size
        self.mask_ratio = mask_ratio
        self.knn = KNN(k=32, transpose_mode=True)

    def __call__(self, pc):
        mask_ratio = torch.tensor([random.uniform(self.mask_ratio[0], self.mask_ratio[1])], device=pc.device)
        bsize = pc.size(0)
        center = misc.fps(pc, self.num_group)  # B G 3
        _, idx = self.knn(pc, center)  # B G M

        num_groups_deleted = (self.num_group * mask_ratio).long()
        num_points_deleted = num_groups_deleted * self.group_size
        idx_to_delete = idx[:, :num_groups_deleted].flatten(1)

        first_point = pc[:, 0].unsqueeze(1).expand(-1, num_points_deleted, -1)
        pc.scatter_(1, idx_to_delete.unsqueeze(2).expand(-1, -1, 3), first_point)

        return pc
