#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import builtins
import math
import os
from icecream import ic

os.environ["WANDB_MODE"] = "offline"

import random
import shutil
import time
import warnings

import moco.builder
from datasets import loader_with_fine
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as data
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import wandb
import json


import utils.utils_iNaturalist as utils_iNaturalist

import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore", category=NumbaPendingDeprecationWarning)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import iNaturalist_2019
from datasets.iNaturalist_2019 import TAX_LEVELS_MAPPING, TAXONOMY_NUM


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument('--dataset', type=str, default='iNaturalist2019', choices=['iNaturalist2019'],
                        help='which dataset should be used to pretrain the model')
parser.add_argument("--data", type=str, help="path to dataset")

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=12,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N", help="number of total epochs to run" #80
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)

parser.add_argument(
    "--img_size", default=224, type=int, metavar="N", help="width & heigh of img" 
)


parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.003,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[60, 90],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=50,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(    
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=1228, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.") #"None"
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=65536, 
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.2, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")



# options for personally design
parser.add_argument('--instance_selection', action="store_true", help='Negative sample sampling strategy')
parser.add_argument('--warmup_epoch',default=10, type=int, help='warm up epoch') 
parser.add_argument('--debug', default=False)
parser.add_argument('--dim', default=128, type=int)
parser.add_argument(
    "--exp_dir", default='./temp_experiment/'
)
parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
parser.add_argument("--hie_level", default='species', choices=['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom'],
                     type=str, metavar="Hierarchical Level", help="given label at which Hierarchical Level",
)




def build_dataloaders(args):
    return getattr(loader_with_fine, args.dataset)(args)


def main():
    args = parser.parse_args()

    
    args.debug = False
    assert args.dataset == 'iNaturalist2019', 'dataset is not iNaturalist2019'

    args.TAX_LEVELS_MAPPING = TAX_LEVELS_MAPPING
    args.TAXONOMY_NUM = TAXONOMY_NUM
    selected_taxonomy = args.hie_level    # species
    args.selected_hie_level = args.TAX_LEVELS_MAPPING[selected_taxonomy]   
    args.category_num = args.TAXONOMY_NUM[selected_taxonomy]
    args.cluster_num_perc =  4100 // args.category_num
    args.num_cluster = args.cluster_num_perc * args.category_num
    args.threshold = 1.5
    args.name_list = [selected_taxonomy + '_' + str(i).zfill(2) for i in range(args.category_num)]
    ic(args.num_cluster)
    args.num_hie_level = len(args.TAXONOMY_NUM)
    ic(args.num_hie_level)

    ic('args.moco_t', args.moco_t)

    #cluster_num_perc
    temp = args.cluster_num_perc
    args.cluster_num_perc = []
    for i in range(args.category_num):
        args.cluster_num_perc.append(temp)


    ###wandb
    args.results_dir = f'arch_[{args.arch}]_data[{args.dataset}]_hie_level_[{args.hie_level}]_epochs[{args.epochs}]_memorysize[{args.moco_k}]_temperature_[{args.moco_t}]'
    args.wandb_id = 'iNaturalist2019'
    if not os.path.exists(args.wandb_id):
        os.mkdir(args.wandb_id)
    if not os.path.exists(f'{args.wandb_id}/{args.results_dir}'):
        os.mkdir(f'{args.wandb_id}/{args.results_dir}')


    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],\
        args,
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
    )
    print(model)

     
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir, exist_ok=True)

    ###############

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    class new_CrossEntropyLoss(nn.Module):
        def __init__(self):
            super(new_CrossEntropyLoss, self).__init__()
            
        def forward(self, x, y):
            x = torch.nn.functional.log_softmax(x, dim=1)
            loss = -y * x
            loss = torch.mean(torch.sum(loss, dim=1), dim=0)
            return loss
    new_criterion = new_CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # train_loader & eval_loader is train dataset, val_loader is test dataset
    train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset = build_dataloaders(args)
    dist.barrier()

    if args.eval:
        embedding_dim = model.module.state_dict()['embedding_dim']
        # cluster_result =  None
        cluster_result_name = f'clusters_{checkpoint["epoch"]-1}'
        cluster_result_path =  os.path.join(os.path.dirname(args.resume), cluster_result_name)
        loc = "cuda:{}".format(args.gpu) 
        cluster_result = torch.load(cluster_result_path, map_location=loc)
        args.cluster_num_perc = cluster_result['cluster_num_perc'][0].cpu().numpy().tolist()
        args.num_cluster = sum(args.cluster_num_perc)
        print('args.cluster_num_perc:', args.cluster_num_perc)
        print('args.num_cluster:', args.num_cluster)

        retrieval_topk, knn_topk = utils_iNaturalist.retrieval(model, val_loader, eval_loader, [1, 2, 5, 10, 50, 100, 200], cluster_result, args, embedding_dim)
        retrieval_top1, retrieval_top2, retrieval_top5, retrieval_top10, retrieval_top50, retrieval_top100, retrieval_top200 = retrieval_topk
        knn_top1, knn_top2, knn_top5, knn_top10, knn_top50, knn_top100, knn_top200 = knn_topk

        
        print(f'R@1: {retrieval_top1:.4f}, R@2: {retrieval_top2:.4f}, R@5: {retrieval_top5:.4f}, R@10: {retrieval_top10:.4f},  R@50: {retrieval_top50:.4f},R@100: {retrieval_top100:.4f}, R@200: {retrieval_top200:.4f}')
        print(f'knn@1: {knn_top1:.4f}, knn@2: {knn_top2:.4f}, knn@5: {knn_top5:.4f}, knn@10: {knn_top10:.4f},  knn@50: {knn_top50:.4f},knn100: {knn_top100:.4f}, knn@200: {knn_top200:.4f}')
        
        exit(0)




    ####wandb
    wandb.init(project='BUEM', entity=args.wandb_id, name='train_' + args.results_dir, group=f'train_{args.dataset}')
    wandb.config.update(args)
    with open(f'{args.wandb_id}/{args.results_dir}' + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)
    train_logs = open(f'{args.wandb_id}/{args.results_dir}/train_logs.txt', 'w')

    best_retrieval_top1 = [0] * args.num_hie_level 
    best_retrieval_top2 = [0] * args.num_hie_level 
    best_retrieval_top5 = [0] * args.num_hie_level 
    best_retrieval_top10 = [0] * args.num_hie_level 
    best_retrieval_top50 = [0] * args.num_hie_level 
    best_retrieval_top100 = [0] * args.num_hie_level 
    best_retrieval_top200 = [0] * args.num_hie_level 

    best_knn_top1 = [0] * args.num_hie_level 
    best_knn_top2 = [0] * args.num_hie_level 
    best_knn_top5 = [0] * args.num_hie_level 
    best_knn_top10 = [0] * args.num_hie_level 
    best_knn_top50 = [0] * args.num_hie_level 
    best_knn_top100 = [0] * args.num_hie_level 
    best_knn_top200 = [0] * args.num_hie_level 

    embedding_dim = model.module.state_dict()['embedding_dim']


    cluster_result = None
    for epoch in range(args.start_epoch, args.epochs):
        print('============================================================================')
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        
        for param_group in optimizer.param_groups:
            lr_check = param_group['lr']         
        print('epoch:{}, lr is:{}'.format(epoch,lr_check))        
        
        # merging clusters
        # cluster_result = None
        if epoch >= args.warmup_epoch:
            features = utils_iNaturalist.compute_features(eval_loader, model, args, is_mlp_k=True, is_embedding_q=False, is_embedding_k=False, embedding_dim=embedding_dim) 
            fea_dim = (features.shape[-1] // 64) * 64


            print("dim of feature: ", features.shape)
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[],'cluster_num_perc':[]}
            num_cluster_tensor = torch.tensor(int(args.num_cluster)).cuda()
            if dist.get_rank() == 0:
                features_cpu = features.cpu().numpy()
                cluster_result = utils_iNaturalist.run_hkmeans(features_cpu, args, fea_dim) 
                
                if (epoch + 1) % 5 == 0:
                    try:
                        torch.save(cluster_result, os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
                    except:
                        pass
                print('args.num_cluster', args.num_cluster)
                num_cluster_tensor = torch.tensor(args.num_cluster).cuda()

            # start_time = time.time()
            dist.barrier()
            dist.broadcast(num_cluster_tensor, src=0, async_op=True)
            if dist.get_rank() != 0:
                args.num_cluster = int(num_cluster_tensor.item())
                print('args.num_cluster', args.num_cluster)
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(args.num_cluster),args.dim).cuda())
                cluster_result['density'].append(torch.zeros(int(args.num_cluster)).cuda())
                cluster_result['cluster_num_perc'].append(torch.zeros(int(args.category_num)).cuda())
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:           
                    dist.broadcast(data_tensor, 0, async_op=False)        


        # train for one epoch
        train(train_loader, model, criterion, new_criterion, optimizer, epoch, args, cluster_result)

        #recall test
        if (epoch + 1) % 10 == 0:
            retrieval_topk_list, knn_topk_list = utils_iNaturalist.retrieval(model, val_loader, eval_loader, [1, 2, 5, 10, 50, 100, 200], args, embedding_dim)

            print('$$$$$$$$$$'*5)
            for idx in range(args.num_hie_level):
                print(f'{idx}-level has retrieval: ')
                retrieval_top1, retrieval_top2, retrieval_top5, retrieval_top10, retrieval_top50, retrieval_top100, retrieval_top200 = retrieval_topk_list[idx]
                print(f'Epoch [{epoch}/{args.epochs}/{idx}-level]: R@1: {retrieval_top1:.4f}, R@2: {retrieval_top2:.4f}, R@5: {retrieval_top5:.4f}, R@10: {retrieval_top10:.4f}, R@50: {retrieval_top50:.4f}, R@100: {retrieval_top100:.4f}, R@200: {retrieval_top200:.4f}')
                train_logs.write(
                f'Epoch [{epoch}/{args.epochs}/{idx}-level]: R@1: {retrieval_top1:.4f}, R@2: {retrieval_top2:.4f}, R@5: {retrieval_top5:.4f}, R@10: {retrieval_top10:.4f}, R@50: {retrieval_top50:.4f}, R@100: {retrieval_top100:.4f}, R@200: {retrieval_top200:.4f}\n')
                train_logs.flush()

                if retrieval_top1 > best_retrieval_top1[idx]:
                    best_retrieval_top1[idx] = retrieval_top1
                if retrieval_top2 > best_retrieval_top2[idx]:
                    best_retrieval_top2[idx] = retrieval_top2
                if retrieval_top5 > best_retrieval_top5[idx]:
                    best_retrieval_top5[idx] = retrieval_top5
                if retrieval_top10 > best_retrieval_top10[idx]:
                    best_retrieval_top10[idx] = retrieval_top10
                if retrieval_top50 > best_retrieval_top50[idx]:
                    best_retrieval_top50[idx] = retrieval_top50
                if retrieval_top100 > best_retrieval_top100[idx]:
                    best_retrieval_top100[idx] = retrieval_top100
                if retrieval_top200 > best_retrieval_top200[idx]:
                    best_retrieval_top200[idx] = retrieval_top200

            print('$$$$$$$$$$'*5)
            for idx in range(args.num_hie_level):
                print(f'{idx}-level has knn:')
                knn_top1, knn_top2, knn_top5, knn_top10, knn_top50, knn_top100, knn_top200 = knn_topk_list[idx]
                print(f'Epoch [{epoch}/{args.epochs}/{idx}-level]: knn@1: {knn_top1:.4f}, knn@2: {knn_top2:.4f}, knn@5: {knn_top5:.4f}, knn@10: {knn_top10:.4f}, knn@50: {knn_top50:.4f}, knn100: {knn_top100:.4f}, knn@200: {knn_top200:.4f}')
                train_logs.write(
                    f'Epoch [{epoch}/{args.epochs}/{idx}-level]: K@1: {knn_top1:.4f}, K@2: {knn_top2:.4f}, K@5: {knn_top5:.4f}, K@10: {knn_top10:.4f}, K@50: {knn_top50:.4f}, K@100: {knn_top100:.4f}, K@200: {knn_top200:.4f}\n')            
                train_logs.flush()
        
        
                if knn_top1 > best_knn_top1[idx]:
                    best_knn_top1[idx] = knn_top1
                if knn_top2 > best_knn_top2[idx]:
                    best_knn_top2[idx] = knn_top2
                if knn_top5 > best_knn_top5[idx]:
                    best_knn_top5[idx] = knn_top5
                if knn_top10 > best_knn_top10[idx]:
                    best_knn_top10[idx] = knn_top10
                if knn_top50 > best_knn_top50[idx]:
                    best_knn_top50[idx] = knn_top50
                if knn_top100 > best_knn_top100[idx]:
                    best_knn_top100[idx] = knn_top100
                if knn_top200 > best_knn_top200[idx]:
                    best_knn_top200[idx] = knn_top200


        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    filename=args.exp_dir + '/' + "checkpoint_{:04d}.pth.tar".format(epoch),
                )
    print('============='*4)
    print(f'[Best]: R@1: {best_retrieval_top1}')
    print(f'[Best]: R@2: {best_retrieval_top2}')
    print(f'[Best]: R@5: {best_retrieval_top5}')
    print(f'[Best]: R@10: {best_retrieval_top10}')
    print(f'[Best]: R@50: {best_retrieval_top50}')
    print(f'[Best]: R@100: {best_retrieval_top100}')
    print(f'[Best]: R@200: {best_retrieval_top200}')
    
    print('============='*4)
    print(f'[Best]: knn@1: {best_knn_top1}')
    print(f'[Best]: knn@2: {best_knn_top2}')
    print(f'[Best]: knn@5: {best_knn_top5}')
    print(f'[Best]: knn@10: {best_knn_top10}')
    print(f'[Best]: knn@50: {best_knn_top50}')
    print(f'[Best]: knn@100: {best_knn_top100}')
    print(f'[Best]: knn@200: {best_knn_top200}')


    train_logs.write(f'[Best]: R@1: {best_retrieval_top1}\n')
    train_logs.write(f'[Best]: R@2: {best_retrieval_top2}\n')
    train_logs.write(f'[Best]: R@5: {best_retrieval_top5}\n')
    train_logs.write(f'[Best]: R@10: {best_retrieval_top10}\n')
    train_logs.write(f'[Best]: R@50: {best_retrieval_top50}\n')
    train_logs.write(f'[Best]: R@100: {best_retrieval_top100}\n')
    train_logs.write(f'[Best]: R@200: {best_retrieval_top200}\n')


    train_logs.write(f'[Best]: knn@1: {best_knn_top1}\n')
    train_logs.write(f'[Best]: knn@2: {best_knn_top2}\n')
    train_logs.write(f'[Best]: knn@5: {best_knn_top5}\n')
    train_logs.write(f'[Best]: knn@10: {best_knn_top10}\n')
    train_logs.write(f'[Best]: knn@50: {best_knn_top50}\n')
    train_logs.write(f'[Best]: knn@100: {best_knn_top100}\n')
    train_logs.write(f'[Best]: knn@200: {best_knn_top200}\n')       
    
    train_logs.flush()

    wandb.finish()



def train(train_loader, model, criterion, new_criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = dict()
    acc_inst = dict()
    losses['Loss_sum'] = AverageMeter('Loss_sum', ':.4e')
    acc_inst['Acc@Inst_avg'] = AverageMeter('Acc@Inst_avg', ':6.2f')
    buffer_meter = dict()
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, buffer_meter],
        prefix="Epoch: [{}]".format(epoch))


    # switch to train mode
    model.train()

    end = time.time()

    for i, (images, index, target_list) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

            labels = target_list[args.selected_hie_level]
            labels = labels.cuda(args.gpu, non_blocking=True)

        # compute output
        output, target, logits, weights = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index, labels_queue=labels, epoch=epoch) 
        

        if isinstance(target, list):
            # print('yes target is a list')
            loss = 0.
            for k, (out, tar) in enumerate(zip(output, target)):
                if weights == None:
                    con_loss_1 = criterion(out, tar)
                else:
                    con_loss_1 = new_criterion(out, weights)
                con_loss_2 = criterion(logits, labels)
                

                acc = accuracy(out, tar)[0] 
                if f'Acc@Inst{k}' not in buffer_meter:
                    buffer_meter[f'Acc@Inst{k}'] = AverageMeter(f'Acc@Inst{k}', ":6.2f")
                buffer_meter[f'Acc@Inst{k}'].update(acc[0], images[0].size(0))
                acc_inst['Acc@Inst_avg'].update(acc[0], images[0].size(0))

                alpha = 0.5
                loss =  loss + alpha * con_loss_1 + (1-alpha) * con_loss_2  

            losses['Loss_sum'].update(loss.item(), images[0].size(0))

        else:
            con_loss_1 = criterion(output, target) 
            con_loss_2 = criterion(logits, labels) 
            alpha = 0.5
            loss =  alpha * con_loss_1 + (1-alpha) * con_loss_2            

            losses['Loss_sum'].update(loss.item(), images[0].size(0))
            acc = accuracy(output, target)[0] 
            acc_inst['Acc@Inst_avg'].update(acc[0], images[0].size(0))            


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if dist.get_rank() == 0:
                progress.display(i)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        for meter in self.meters:
            if isinstance(meter, AverageMeter):
                entries += [str(meter)]
            elif isinstance(meter, dict):
                entries += [str(v) for (k, v) in meter.items()]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
