# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function

from PIL import ImageFilter
import random
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


import os
import os.path


from datasets import *


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


# # one strong and one weak augmentation
#     def __init__(self, strong_aug, weak_aug):
#         self.weak_transform = weak_aug
#         self.strong_transform = strong_aug

#     def __call__(self, x):
#         q = self.weak_transform(x)
#         k = self.strong_transform(x)
#         return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index, target, index  


def build_augmentation(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    fig_size = args.img_size
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(fig_size, scale=(0.25, 1.0)),  
            transforms.RandomHorizontalFlip(),  
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.8), 
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(fig_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    augmentation = TwoCropsTransform(transforms.Compose(augmentation))

    ratio = 224 / 256
    before_size = int(fig_size / ratio)
    eval_augmentation = transforms.Compose(
        [
            transforms.Resize(before_size),
            transforms.CenterCrop(fig_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return augmentation, eval_augmentation


def Travel(args):
    # Data loading code
    if args.debug:
        traindir = os.path.join(args.data, "val")
    else:
        traindir = os.path.join(args.data, "trainval")
    augmentation, eval_augmentation = build_augmentation(args)

    train_dataset = ImageFolderInstance(traindir, augmentation)
    eval_dataset = ImageFolderInstance(traindir, eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, None, None


def place20(args):  # 80000
    # Data loading code
    if args.debug:
        traindir = os.path.join(args.data, "test")
    else:
        traindir = os.path.join(args.data, "trainval")
    augmentation, eval_augmentation = build_augmentation(args)

    train_dataset = ImageFolderInstance(traindir, augmentation)
    eval_dataset = ImageFolderInstance(traindir, eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, None, None


def ILSVRC20(args):  # 20000
    # Data loading code
    if args.debug:
        traindir = os.path.join(args.data, "test")
    else:
        traindir = os.path.join(args.data, "trainval")
    augmentation, eval_augmentation = build_augmentation(args)

    train_dataset = ImageFolderInstance(traindir, augmentation)
    eval_dataset = ImageFolderInstance(traindir, eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, None, None


def cifar100(args):
    fig_size = args.img_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation, eval_augmentation = build_augmentation(args)

    train_dataset = CIFAR100(root=args.data, train=True, download=True, transform=augmentation)
    eval_dataset = CIFAR100(root=args.data, train=True, download=True, transform=eval_augmentation)
    val_dataset = CIFAR100(root=args.data, train=False, download=True, transform=eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def cifar_toy(args):
    fig_size = args.img_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation, eval_augmentation = build_augmentation(args)

    if args.good_split:
        train_dataset = CIFARtoy(root=args.data, split="good", download=True, transform=augmentation)
        eval_dataset = CIFARtoy(root=args.data, split="good", train=True, download=True, transform=eval_augmentation)
        val_dataset = CIFARtoy(root=args.data, split="good", train=False, download=True, transform=eval_augmentation)
    else:
        train_dataset = CIFARtoy(root=args.data, split="bad", download=True, transform=augmentation)
        eval_dataset = CIFARtoy(root=args.data, split="bad", train=True, download=True, transform=eval_augmentation)
        val_dataset = CIFARtoy(root=args.data, split="bad", train=False, download=True, transform=eval_augmentation)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def cifartoy_good(args):
    fig_size = args.img_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation, eval_augmentation = build_augmentation(args)


    train_dataset = CIFARtoy(root=args.data, split="good", download=True, transform=augmentation)
    eval_dataset = CIFARtoy(root=args.data, split="good", train=True, download=True, transform=eval_augmentation)
    val_dataset = CIFARtoy(root=args.data, split="good", train=False, download=True, transform=eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def cifartoy_bad(args):
    fig_size = args.img_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation, eval_augmentation = build_augmentation(args)

    train_dataset = CIFARtoy(root=args.data, split="bad", download=True, transform=augmentation)
    eval_dataset = CIFARtoy(root=args.data, split="bad", train=True, download=True, transform=eval_augmentation)
    val_dataset = CIFARtoy(root=args.data, split="bad", train=False, download=True, transform=eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset



def imagenet(args):
    augmentation, eval_augmentation = build_augmentation(args)
    train_dataset = ImageNetDownSample(root=args.data, train=True, transform=augmentation)
    eval_dataset = ImageNetDownSample(root=args.data, train=True, transform=eval_augmentation)
    val_dataset = ImageNetDownSample(root=args.data, train=False, transform=eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def iNaturalist2019(args):

    query_transform = get_augment(args.dataset, "strong", size=args.img_size)
    key_transform = get_augment(args.dataset, "weak", size=args.img_size)
    augmentation = DMixTransform([key_transform, query_transform], [1, 1])
    eval_augmentation = get_augment(args.dataset, size=args.img_size)

    train_dataset = iNaturalist_2019(root=args.data, mode="trainval", transform=augmentation)
    eval_dataset = iNaturalist_2019(root=args.data, mode="trainval", transform=eval_augmentation)
    val_dataset = iNaturalist_2019(root=args.data, mode="test", transform=eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers // 2, persistent_workers=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers // 4, persistent_workers=True, pin_memory=True
    )
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def aircraft(args):
    query_transform = get_augment(args.dataset, "strong", size=args.img_size)
    key_transform = get_augment(args.dataset, "weak", size=args.img_size)
    augmentation = DMixTransform([key_transform, query_transform], [1, 1])
    eval_augmentation = get_augment(args.dataset, size=args.img_size)

    train_dataset = FGVCAircraft(root=args.data, split="trainval", transform=DMixTransform([key_transform, query_transform], [1, 1]))
    eval_dataset = FGVCAircraft(root=args.data, split="trainval", transform=eval_augmentation)
    val_dataset = FGVCAircraft(root=args.data, split="test", transform=eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def flowers102(args):
    query_transform = get_augment(args.dataset, "strong", size=args.img_size)
    key_transform = get_augment(args.dataset, "weak", size=args.img_size)
    augmentation = DMixTransform([query_transform, query_transform], [1, 1])
    eval_augmentation = get_augment(args.dataset, size=args.img_size)

    # args.data = "/media/ssd2T/Datasets/flowers102"
    train_dataset = MyFlowers102(root=args.data, split="train", transform=augmentation)
    eval_dataset = MyFlowers102(root=args.data, split="train", transform=eval_augmentation)
    val_dataset = MyFlowers102(root=args.data, split="test", transform=eval_augmentation)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset

def cars196(args):
    query_transform = get_augment(args.dataset, "strong", size=args.img_size)
    key_transform = get_augment(args.dataset, "weak", size=args.img_size)
    augmentation = DMixTransform([key_transform, query_transform], [1, 1])
    eval_augmentation = get_augment(args.dataset, size=args.img_size)

    train_dataset = CARS196(root=args.data, split="train", transform=augmentation)
    eval_dataset = CARS196(root=args.data, split="train", transform=eval_augmentation)
    val_dataset = CARS196(root=args.data, split="test", transform=eval_augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset

def nabirds(args):
    query_transform = get_augment(args.dataset, "strong", size=args.img_size)
    key_transform = get_augment(args.dataset, "weak", size=args.img_size)
    augmentation = DMixTransform([query_transform, query_transform], [1, 1])
    eval_augmentation = get_augment(args.dataset, size=args.img_size)

    train_dataset = MyCustomNabirds(root=args.data, split="train", transform=augmentation)
    eval_dataset = MyCustomNabirds(root=args.data, split="train", transform=eval_augmentation)
    val_dataset = MyCustomNabirds(root=args.data, split="test", transform=eval_augmentation)



    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def cub200(args):
    query_transform = get_augment(args.dataset, "strong", size=args.img_size)
    key_transform = get_augment(args.dataset, "weak", size=args.img_size)
    augmentation = DMixTransform([query_transform, query_transform], [1, 1])
    eval_augmentation = get_augment(args.dataset, size=args.img_size)

    train_dataset = CUB200(path=args.data, train=True, transform=augmentation)
    eval_dataset = CUB200(path=args.data, train=True, transform=eval_augmentation)
    val_dataset = CUB200(path=args.data, train=False, transform=eval_augmentation)


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        drop_last=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    
    return train_loader, eval_loader, train_dataset, eval_dataset, train_sampler, val_loader, val_dataset


def build_augmentation_iNaturalist(args):

    normalize = transforms.Normalize(mean=[0.454, 0.474, 0.367], std=[0.237, 0.230, 0.249])
    fig_size = args.img_size
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(fig_size, scale=(0.25, 1.0)),  # best
            transforms.RandomHorizontalFlip(),  
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  
            transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)], p=0.8),  
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(fig_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    augmentation = TwoCropsTransform(transforms.Compose(augmentation))

    ratio = 224 / 256
    before_size = int(fig_size / ratio)
    eval_augmentation = transforms.Compose(
        [
            transforms.Resize(before_size),
            transforms.CenterCrop(fig_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return augmentation, eval_augmentation


class DMixTransform:
    def __init__(self, transforms, nums):
        self.transforms = transforms
        self.nums = nums

    def __call__(self, x):
        res = []
        for i, trans in enumerate(self.transforms):
            res += [trans(x) for _ in range(self.nums[i])]
        return res


def get_augment(dataset, mode="none", size=224):

    if dataset == "cifartoy_good" or dataset == "cifartoy_bad":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = size
        # extra = CIFAR10Policy()
    elif dataset == "cifar100":
        mean = (0.507, 0.487, 0.441)
        std = (0.267, 0.256, 0.276)
        size = size
        # extra = CIFAR10Policy()
    elif dataset == "cars196":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = size
    elif dataset == "aircraft":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = size
    elif dataset == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = size
    elif dataset == "iNaturalist2019":
        mean = (0.454, 0.474, 0.367)
        std = (0.237, 0.230, 0.249)
        size = size
    elif dataset == "flowers102":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = size
    elif dataset == "nabirds":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = size
    elif dataset == "cub200":
        mean = (0.486, 0.499, 0.432)
        std = (0.232, 0.228, 0.267)
        size = size

    else:
        raise ValueError(f"dataset should not be {dataset}!")
    normalize = transforms.Normalize(mean=mean, std=std)

    if mode == "strong":
        if dataset == "cars196" or dataset == "iNaturalist2019" or dataset == "aircraft" or dataset == "flowers102" or dataset == "nabirds" or dataset == "cub200":
            res_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=[size, size], scale=(0.2, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomPerspective(0.5, 0.5),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    elif mode == "weak":
        if dataset == "cars196" or dataset == "iNaturalist2019" or dataset == "aircraft" or dataset == "flowers102" or dataset == "nabirds" or dataset == "cub200":
            res_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=[size, size], scale=(0.2, 1)),
                    transforms.RandomPerspective(0.5, 0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            res_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=[size, size], scale=(0.2, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    else:  # mode == 'test'/'none'
        ratio = 224 / 256
        before_size = int(size / ratio)
        res_transform = transforms.Compose(
            [
                transforms.Resize(before_size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return res_transform
