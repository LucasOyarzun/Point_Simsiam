import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils import misc
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
class PointSimsiam(nn.Module):
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


# finetune model
@MODELS.register_module()
class PointSimsiamClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_dim = config.cls_dim
        self.backbone = PointNetfeat()
        self.projector = projection_MLP(self.backbone.output_dim)
        self.encoder = nn.Sequential(self.backbone, self.projector)
        self.predictor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.cls_dim)
            )
        
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            # for k in list(base_ckpt.keys()):
            #     if k.startswith('projector'):
            #         base_ckpt[k[len('projector.'):]] = base_ckpt[k]
            #         del base_ckpt[k]

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
        x = x.permute(0, 2, 1)
        f = self.encoder
        p = self.predictor
        z = nn.Sequential(f, p)(x)
        return z

