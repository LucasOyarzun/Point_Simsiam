import torch
from tools import builder
import visdom
from utils.visdom import vis_pc
import numpy as np
from datasets import data_transforms
from sklearn.manifold import TSNE
from datasets import data_transforms
from torchvision import transforms
from extensions.chamfer_dist import ChamferDistanceL2
from pointnet2_ops import pointnet2_utils
from utils import misc


test_invariation_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudJitter(),
        data_transforms.PointcloudScaleAndTranslate(),
        data_transforms.PointCloudMask(mask_ratio=[0.5, 0.8]),
    ]
)

# %% To test: run in terminal: python -m visdom.server -p 8099
#             python main.py --test_invariation --exp_name testeando --ckpts experiments/pretrain/PointBYOL/byol_pointmlp_linear_mask_05_08/ckpt-last.pth --config cfgs/PointBYOL/finetune_modelnet.yaml 
def run_test_invariation(args, config):
    vis = visdom.Visdom(
        port=8099,
    )
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:    
        base_model.to(args.local_rank)
    

    clouds_generated = 0
    batch_generated = 7
    batch = 0
    with torch.no_grad():
        base_model.eval()
        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            # Pick 5 random values from data
            points = data[0].cuda()
            labels = data[1].cuda()
            points = misc.fps(points, npoints)
            original_points = points.clone()

            transformed_points = test_invariation_transforms(points)
            original_points = original_points.transpose(1, 2).contiguous()
            transformed_points = transformed_points.transpose(1, 2).contiguous()
            feats = base_model.encoder.forward(original_points)
            transformed_feats = base_model.encoder.forward(transformed_points)
            
            if idx == 1:
                original_points = original_points.transpose(1, 2).contiguous()
                transformed_points = transformed_points.transpose(1, 2).contiguous()
                vis_pc(vis, original_points[0], 'Original')
                vis_pc(vis, transformed_points[0], 'Transformed')
                chamfer_dist = ChamferDistanceL2().cuda()
                print("Chamfer distance (Points cloud):", chamfer_dist(original_points[0], transformed_points[0]))
                print("features similarity (Features):", torch.nn.functional.cosine_similarity(feats[0], transformed_feats[0], dim=0))
            else:
                return

