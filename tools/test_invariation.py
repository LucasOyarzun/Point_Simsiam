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
        # data_transforms.PointcloudRotate(axis=0),
        # data_transforms.PointcloudRotate(axis=1),
        # data_transforms.PointcloudRotate(axis=2),
        data_transforms.PointCloudMask(mask_ratio=[0.8, 0.8]),
    ]
)

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
            feats = base_model.encoder.forward(original_points)
            transformed_feats = base_model.encoder.forward(transformed_points)
            
            batch += 1
            if batch == batch_generated:
                idxs = np.random.randint(0, len(data), 5)
                for i in idxs:
                    clouds_generated += 1
                    vis_pc(vis, original_points[i], 'Original')
                    vis_pc(vis, transformed_points[i], 'Transformed')
                    chamfer_dist = ChamferDistanceL2().cuda()
                    print("Chamfer distance (Points cloud):", chamfer_dist(original_points[i], transformed_points[i]))
                    print("features similarity (Features):", torch.nn.functional.cosine_similarity(feats[i], transformed_feats[i], dim=0))
                if clouds_generated == 5:
                    break
            else: pass
