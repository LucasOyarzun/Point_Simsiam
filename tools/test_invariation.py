import torch
from tools import builder
from datasets import data_transforms
from torchvision import transforms
from extensions.chamfer_dist import ChamferDistanceL2
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

def run_test_invariation(args, config):
    # build dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)

    # build model
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:    
        base_model.to(args.local_rank)
    
    with torch.no_grad():
        base_model.eval()
        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            points = data[0].cuda()
            points = misc.fps(points, npoints)
            original_points = points.clone()
            transformed_points = test_invariation_transforms(points)
            if idx == 1:
                original_points = original_points.transpose(1, 2).contiguous()
                transformed_points = transformed_points.transpose(1, 2).contiguous()
                chamfer_dist = ChamferDistanceL2().cuda()
                print("Chamfer distance (Points cloud):", chamfer_dist(original_points[0], transformed_points[0]))
            else:
                return
