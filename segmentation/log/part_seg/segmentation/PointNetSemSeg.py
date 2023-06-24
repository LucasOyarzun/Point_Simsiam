import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import sys
sys.path.append("../utils")
sys.path.append("..")
sys.path.append("./")
from pointnet_utils import STN3d, STNkd
from checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, space_transform=False, feature_transform=False):
        # Feature Transform for PointNet++
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.output_dim = 1024
        self.global_feat = global_feat
        self.space_transform = space_transform
        self.feature_transform = feature_transform
        if self.space_transform:
            self.stn = STN3d()
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[1]
        if self.space_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetSemSeg(nn.Module):
    def __init__(self, cls_dim):
        super(PointNetSemSeg, self).__init__()
        self.cls_dim = cls_dim
        self.encoder = PointNetEncoder(global_feat=False, feature_transform=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.cls_dim, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def load_model_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        incompatible = self.load_state_dict(state_dict['base_model'], strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='Point_M2AE_ModelNet40')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Point_M2AE_ModelNet40'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Point_M2AE_ModelNet40')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Point_M2AE_ModelNet40'
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = self.encoder(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x
    
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss