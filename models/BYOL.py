import copy
import torch 
from torch import nn 
from math import pi, cos 
from timm.models.layers import DropPath, trunc_normal_
from .build import MODELS
from .common.LinearClassifier import LinearClassifier
from .backbones import *
from .Simsiam import D  # a bit different but it's essentially the same thing: neg cosine sim & stop gradient
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class prediction_MLP(nn.Module):
    def __init__(
        self, in_dim=256, hidden_dim=4096, out_dim=256
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
class PointBYOL(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = PointNetfeat()
        self.projector = projection_MLP(self.encoder.output_dim)
        self.online_encoder = nn.Sequential(self.encoder, self.projector)
        self.online_predictor = prediction_MLP()
        self.target_encoder = nn.Sequential(self.encoder, self.projector)

    @classmethod
    def target_ema(cls, k, K, tau_base=0.996):
        return 1 - tau_base * (cos(pi*k/K)+1)/2 

    def update_moving_average(self, global_step, max_steps):
        tau = self.target_ema(global_step, max_steps)
        self.target_encoder.eval()
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    def forward(self, x1, x2):
        f_o, h_o = self.online_encoder, self.online_predictor
        f_t = self.target_encoder

        z1_o = f_o(x1)
        z2_o = f_o(x2)

        p1_o = h_o(z1_o)
        p2_o = h_o(z2_o)

        with torch.no_grad():
            z1_t = f_t(x1)
            z2_t = f_t(x2)
        
        L = D(p1_o, z2_t) / 2 + D(p2_o, z1_t) / 2 
        return L
    
# finetune model
@MODELS.register_module()
class PointBYOLClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_dim = config.cls_dim
        self.encoder = PointNetfeat()
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
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
                

    def forward(self, x, eval_encoder=False):
        b = self.encoder
        if eval_encoder: # for linear svm
            return b(x)
        
        c = self.cls_head_finetune
        z = nn.Sequential(b, c)(x)
        return z