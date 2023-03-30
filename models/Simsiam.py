import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import *
from .build import MODELS


def D(p, z, version="simplified"):
    if version == "original":
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
    elif version == "simplified":
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        """ page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        """
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 2

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(
        self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

@MODELS.register_module()
class Simsiam(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = PointNetfeat()
        self.projector = projection_MLP(self.backbone.output_dim)

        self.encoder = nn.Sequential(self.backbone, self.projector)  # f encoder
        self.predictor = prediction_MLP()

    def forward(self, x1, x2):

        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        loss = D(p1, z2) / 2 + D(p2, z1) / 2
        return loss

    def forward_finetune(self, x):
        print(x.shape)
        f = self.encoder
        print(f(x).shape)
        p = self.predictor
        print(p(x).shape)
        z = nn.Sequential(f, p)(x)
        return z


if __name__ == "__main__":
    model = SimSiam()
    x1 = torch.randn(2, 1024, 3).transpose(2, 1)
    x2 = torch.rand_like(x1)
    print(x1.shape)
    print(x2.shape)
    output = model.backbone(x1)
    print(output.shape)
