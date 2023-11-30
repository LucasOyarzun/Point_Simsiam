import torch
from tools import builder
from utils.logger import *
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go
import plotly.io as pio
import time
import torch.nn as nn

# conda install -c conda-forge umap-learn
def run_tsne_umap(args, config):
    NUM_CLASSES = 40
    classes_names = json.load(open('data/ModelNet/modelnet40_normal_resampled/modelnet40_classes.json', 'r'))
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    # build model
    base_model = builder.model_builder(config.model)

    # resume ckpts
    base_model.load_model_from_ckpt(args.ckpts)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    X, y = [], []
    # with torch.no_grad():
    #     base_model.eval()
    #     for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
    #         points = data[0].cuda()
    #         label = data[1].cuda()
    #         feats = base_model(points)
    #         y.append(label.cpu().numpy())
    #         X.append(feats.cpu().numpy())

    X, y = np.concatenate(X), np.concatenate(y)
    X_embedded = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=0).fit_transform(X)
    plt.figure(figsize=(16, 16))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.get_cmap("jet", NUM_CLASSES))
    #plt.colorbar(ticks=range(1, NUM_CLASSES + 1))
    plt.clim(0.5, NUM_CLASSES + 0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(np.arange(1, len(classes_names) + 1))
    cbar.set_ticklabels(classes_names)
    plt.savefig(f'figures/{args.exp_name}_umap.pdf')

    X_embedded = TSNE(n_components=2, perplexity=100, random_state=0).fit_transform(X)
    plt.figure(figsize=(16, 16))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=plt.cm.get_cmap("jet", NUM_CLASSES))
    #plt.colorbar(ticks=range(1, NUM_CLASSES + 1))
    plt.clim(0.5, NUM_CLASSES + 0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_ticks(np.arange(1, len(classes_names) + 1))
    cbar.set_ticklabels(classes_names)
    plt.savefig(f'figures/{args.exp_name}_tsne.pdf')

    # else:
    #     # UMAP visualization
    #     X_embedded_umap = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=3, random_state=0).fit_transform(X)
    #     fig_umap = go.Figure(data=go.Scatter3d(
    #         x=X_embedded_umap[:, 0],
    #         y=X_embedded_umap[:, 1],
    #         z=X_embedded_umap[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=3,
    #             color=y,
    #             colorscale='Jet',
    #             opacity=0.8
    #         )
    #     ))
    #     fig_umap.update_layout(scene=dict(
    #         xaxis_title='UMAP 1',
    #         yaxis_title='UMAP 2',
    #         zaxis_title='UMAP 3'
    #     ))
    #     pio.write_html(fig_umap, f'figures/{args.exp_name}_umap.html')
        
    #     # t-SNE visualization
    #     X_embedded_tsne = TSNE(n_components=3, perplexity=100, random_state=0).fit_transform(X)
    #     fig_tsne = go.Figure(data=go.Scatter3d(
    #         x=X_embedded_tsne[:, 0],
    #         y=X_embedded_tsne[:, 1],
    #         z=X_embedded_tsne[:, 2],
    #         mode='markers',
    #         marker=dict(
    #             size=3,
    #             color=y,
    #             colorscale='Jet',
    #             opacity=0.8
    #         )
    #     ))
    #     fig_tsne.update_layout(scene=dict(
    #         xaxis_title='t-SNE 1',
    #         yaxis_title='t-SNE 2',
    #         zaxis_title='t-SNE 3'
    #     ))
    #     pio.write_html(fig_tsne, f'figures/{args.exp_name}_tsne.html')
