import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
import umap

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run_visualization(args, config, train_writer=None, val_writer=None):
    NUM_CLASSES = 40
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)


    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Visualizating scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)

    X_train, y_train, X_test, y_test = [], [], [], []
    with torch.no_grad():
        base_model.eval()
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):

            points = data[0].cuda()
            label = data[1].cuda()
            points = points.transpose(2, 1).contiguous() #TODO: Check this
            feats = base_model.encoder.forward(points)
            X_train.append(feats.cpu().numpy())
            y_train.append(label.cpu().numpy())
            
    X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
    X_embedded = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=0).fit_transform(X_train)
    plt.figure(figsize=(16, 16))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train, cmap=plt.cm.get_cmap("jet", NUM_CLASSES))
    plt.colorbar(ticks=range(1, NUM_CLASSES + 1))
    plt.clim(0.5, NUM_CLASSES + 0.5)
    plt.savefig('figures/umap_{args.exp_name}.pdf')

    X_embedded = TSNE(n_components=2, perplexity=100, random_state=0).fit_transform(X_train)
    plt.figure(figsize=(16, 16))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train, cmap=plt.cm.get_cmap("jet", NUM_CLASSES))
    plt.colorbar(ticks=range(1, NUM_CLASSES + 1))
    plt.clim(0.5, NUM_CLASSES + 0.5)
    plt.savefig('figures/tsne_{args.exp_name}.pdf')
