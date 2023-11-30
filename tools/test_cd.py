import torch
from tqdm import tqdm
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
        data_transforms.PointCloudMask(mask_ratio=0.95),
    ]
)

def run_test_cd(args, config):
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
        distances = []
        for idx, (taxonomy_ids, model_ids, data) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            points = data.cuda()
            original_points = points.clone()
            transformed_points = test_invariation_transforms(points)
            chamfer_dist = ChamferDistanceL2().cuda()
            distances.append(chamfer_dist(original_points, transformed_points).item())

        print("Mean Chamfer distance on dataset:", torch.mean(torch.FloatTensor(distances)))
