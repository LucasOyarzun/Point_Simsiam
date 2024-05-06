from tools import builder
from datasets import data_transforms
from torchvision import transforms
from utils import misc
import visdom
import numpy as np

# Point Simsiam
linear_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudJitter(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)
mask_transforms = transforms.Compose(
    [
        data_transforms.PointCloudMask(mask_ratio=0.6),
    ]
)


def vis_pc(vis, pc, title):
    vis.scatter(
        X=pc,
        opts=dict(
            markersize=2,
            title=title,
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
        ),
    )


# To use:
# run in terminal: python -m visdom.server -p 8099
# python main.py --test_transformations --config cfgs/PointSimsiam/pretrain.yaml
def run_test_transformations(args, config):
    # Imprimir la semilla actual
    vis = visdom.Visdom(
        port=8098,
    )
    (train_sampler, train_dataloader) = builder.dataset_builder(
        args, config.dataset.train
    )

    npoints = 1024
    vis_idx = 16
    for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
        if idx == vis_idx:
            points = data[0].cuda()
            points = points.unsqueeze(0)
            points = misc.fps(points, npoints)
            original_points = points.clone()
            vis_pc(vis, original_points[0], "Original")
            transformed_points = linear_transforms(points.clone())
            vis_pc(vis, transformed_points[0], "Transformed")
            transformed_points_2 = mask_transforms(transformed_points.clone())
            vis_pc(vis, transformed_points_2[0], "Transformed 2")
            # save as obj
            np.save("original_points.npy", original_points[0].cpu().numpy())
            np.save("transformed_points.npy", transformed_points[0].cpu().numpy())
            np.save("transformed_points_2.npy", transformed_points_2[0].cpu().numpy())
            return
            # transformed_points_2 = my_transforms(original_points)
            # vis_pc(vis, transformed_points_2[0], "Transformed 2")
