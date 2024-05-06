# Generate with linear probing configs

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import builder
from utils.logger import *


def run_generate_feature_distances(args, config):
    classes_names = json.load(
        open("data/ModelNet/modelnet40_normal_resampled/modelnet40_classes.json", "r")
    )
    logger = get_logger(args.log_name)

    (test_sampler, test_dataloader) = builder.dataset_builder(args, config.dataset.test)
    base_model = load_model(args, config, logger)

    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    X, y = extract_features(base_model, test_dataloader)

    distance_matrix = calculate_distance_matrix(X)
    np.savetxt(
        # f"distance_{config.model.NAME}_{config.model.encoder.NAME}.txt",
        f"distance_{config.model.NAME}.txt",
        distance_matrix.numpy(),
    )

    classes_names = {name: idx for idx, name in enumerate(classes_names)}
    save_classifications(
        # classes_names, y, f"{config.model.NAME}_{config.model.encoder.NAME}.cla"
        classes_names,
        y,
        f"{config.model.NAME}.cla",
    )
    return distance_matrix


def use_data_parallel(args, model, logger):
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            print_log("Using Synchronized BatchNorm ...", logger=logger)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True,
        )
        print_log("Using Distributed Data parallel ...", logger=logger)
    else:
        print_log("Using Data parallel ...", logger=logger)
        model = nn.DataParallel(model).cuda()
    return model


def load_model(args, config, logger):
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts)
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model = use_data_parallel(args, base_model, logger)
    return base_model


def extract_features(model, dataloader):
    X, y = [], []
    with torch.no_grad():
        model.eval()
        for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            feats = model(points)
            y.append(label.cpu().numpy())
            X.append(feats.cpu().numpy())

    X = torch.cat([torch.tensor(x) for x in X], dim=0)
    y = torch.cat([torch.tensor(y) for y in y], dim=0)
    return X, y


def calculate_distance_matrix(X):
    X_norm = F.normalize(
        X, p=2, dim=1
    )  # Normalize feature vectors for cosine similarity
    cosine_similarity_matrix = torch.mm(X_norm, X_norm.transpose(0, 1))
    distance_matrix = 1 - cosine_similarity_matrix
    return distance_matrix


def save_classifications(classes, y, output_file):
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    # Count the number of elements in each class
    class_counts = {class_name: 0 for class_name in classes}
    idxs_per_class = {class_name: [] for class_name in classes}
    # Get the indices of each element in each class
    for i, label in enumerate(y):
        class_name = [name for name, index in classes.items() if index == label][0]
        class_counts[class_name] += 1
        idxs_per_class[class_name].append(str(i))

    # Write the file
    with open(output_file, "w") as file:
        file.write("PSB 1\n")
        file.write(f"{len(classes)} {len(y)}\n\n")

        for class_name, class_index in classes.items():
            file.write(f"{class_name} 0 {class_counts[class_name]}\n")
            for idx in idxs_per_class[class_name]:
                file.write(idx + "\n")
            file.write("\n")
