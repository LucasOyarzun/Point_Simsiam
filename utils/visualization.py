import polyscope as ps
import random, json
from datasets import ShapeNet55Dataset, ScanObjectNNDataset, ModelNetDataset
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from torchvision import transforms
from datasets import data_transforms
def visualize(args, config):
    ps.init()
    # if args.visualization == "shapenet":
        # Load data
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)

    idx = random.randint(0, len(dataset))
    ps.register_point_cloud("Transform 1", dataset[idx][2], enabled=False)
    ps.register_point_cloud("Transform 2", dataset[idx][3], enabled=False)
    ps.register_point_cloud("Original", dataset[idx][4])

    # Print the class of the cloud
    with open("data\\ShapeNet55-34\\Shapenet_classes.json") as json_file:
        ShapeNET_classes = json.load(json_file)
        ps.info(f"Class: {ShapeNET_classes[dataset[idx][0]]}")

    # if args.visualization == "scanobjectnn":
    #     dataset = ScanObjectNNDataset(config=args, split="train")
    #     idx = random.randint(0, len(dataset))

    #     ps.register_point_cloud("Point Cloud", dataset[idx][2][0])
    #     # Print the class of the cloud
    #     point_class = dataset[idx][2][1]
    #     with open("data\\ScanObjectNN\\scanobjectnn_classes.json") as json_file:
    #         ScanObjectNN_classes = json.load(json_file)["classes"]
    #         ps.info(f"Class: {ScanObjectNN_classes[point_class]}")

    # if args.visualization == "modelnet":
    #     dataset = ModelNetDataset(config=args, npoints=8192, split="train")
    #     idx = random.randint(0, len(dataset))
    #     ps.register_point_cloud("Point Cloud", dataset[idx][2][0])
    #     # Print the class of the cloud
    #     point_class = dataset[idx][2][1]
    #     with open(
    #         "data\\ModelNet\\modelnet40_normal_resampled\\modelnet40_classes.json"
    #     ) as json_file:
    #         ScanObjectNN_classes = json.load(json_file)["classes"]
    #         ps.info(f"Class: {ScanObjectNN_classes[point_class]}")

    ps.show()


if __name__ == "__main__":
    # args = parse_args()
    visualize_point_cloud()
