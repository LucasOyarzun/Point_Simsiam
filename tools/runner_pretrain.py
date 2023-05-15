import torch
import torch.nn as nn
import json
import random
import visdom

from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from utils.visdom import vis_pc
from torchvision import transforms
from datasets import data_transforms



train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudJitter(),
        data_transforms.PointcloudScaleAndTranslate(),
        data_transforms.PointCloudMask(mask_ratio=[0.5, 0.8]),
    ]
)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    # vis = visdom.Visdom()
    logger = get_logger(args.log_name)

    # build dataset for pre-training
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # training
    base_model.zero_grad()
    load_clouds = 0
    best_accuracy = 0.0
    max_steps = config.max_epoch * len(train_dataloader)
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME

            # Siamese networks requires two different data augmentations
            # if config.dataset.train.others.siamese_network:
            #     [ data1, data2 ] = data
            #     if dataset_name == 'ShapeNet':
            #         # if load_clouds <= 4:
            #         #     img_id = random.randint(0, data1.size(0) - 1)
            #         #     with open("data/ShapeNet55-34/Shapenet_classes.json") as classes_file:
            #         #         classes = json.load(classes_file)
            #         #         vis_pc(vis, data1[img_id], f"{classes[taxonomy_ids[img_id]]}[{img_id}] - 1")
            #         #         vis_pc(vis, data2[img_id], f"{classes[taxonomy_ids[img_id]]}[{img_id}] - 2")
            #         load_clouds += 1
            #         data1 = data1.cuda()
            #         data2 = data2.cuda()
            #     else:
            #         raise NotImplementedError(f'Train phase do not support {dataset_name} for siamese networks')
            #     assert data1.size(1) == npoints
            #     assert data2.size(1) == npoints
            #     data1 = train_transforms(data1)
            #     data2 = train_transforms(data2)
            #     loss = base_model(data1, data2)
                
            # # Other kinds of networks only require one sample with data augmentation
            # else:
            if dataset_name == 'ShapeNet':
                points = data.cuda()
                
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            assert points.size(1) == npoints

            if config.dataset.train.others.siamese_network:
                data1 = points.clone()
                data1 = train_transforms(data1)
                data2 = train_transforms(points)
                assert data1.size(1) == npoints
                assert data2.size(1) == npoints
                data1 = data1.permute(0, 2, 1)
                data2 = data2.permute(0, 2, 1)
                loss = base_model(data1, data2)
            else:
                points = train_transforms(points)
                points = points.permute(0, 2, 1)
                loss = base_model(points)

            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                if config.model.NAME == 'PointBYOL':
                    global_step = epoch * len(train_dataloader) + idx
                    base_model.module.update_moving_average(global_step, max_steps)
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()])
            else:
                losses.update([loss.item()])


            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
            
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
    
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
