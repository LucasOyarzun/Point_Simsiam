import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message


class PointNet2Encoder(nn.Module):
    def __init__(self, normal_channel=True):
        super(PointNet2Cls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, xyz):
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points
    
class PointNet2Cls(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(PointNet2Cls, self).__init__()
        self.normal_channel = normal_channel
        self.encoder = PointNet2Encoder(normal_channel)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

        self.build_loss_func()

    def forward(self, xyz):
        B, _, _ = xyz.shape
        x = self.encoder(xyz)
        x = x.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Classifier')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Classifier'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Classifier')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Classifier'
                )
        else:
            print_log('Training from scratch!!!', logger='Classifier')
            self.apply(self._init_weights)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100
