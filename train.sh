


# ==================== args of training ========================
# for cifar100
CUDA_VISIBLE_DEVICES=0,3 python main.py --dist-url tcp://localhost:10009 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset cifar100 --arch resnet50 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/lxy/data 
--exp_dir ./experiment/cifar100 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos 
| tee -a zz_cifar100.log 

# ====================
# for cifar100 with arch vit
# please revised the path of pretrained vit model
# the pretrained vit model can be download from https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
CUDA_VISIBLE_DEVICES=2,3 python main.py --dist-url tcp://localhost:10009 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset cifar100 --arch vit_base --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/lxy/data 
--exp_dir ./experiment/cifar100_vit_base 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos 
| tee -a zz_cifar100_tmp_vit_base.log 

CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url tcp://localhost:10010 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset cifar100 --arch vit_small --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/lxy/data 
--exp_dir ./experiment/cifar100_vit_small 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos 
| tee -a zz_cifar100_vit_small.log 
# ====================


# for cifar_toy
CUDA_VISIBLE_DEVICES=2,3 python main.py --dist-url tcp://localhost:10009 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset cifartoy_bad --arch resnet50 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/datasets/cifar_toy 
--exp_dir ./experiment/cifar_toy_bad 
--warmup_epoch 10 --epochs 100 --workers 4 --mlp --aug-plus --cos 
| tee -a zz_cifartoy_bad.log 


# for imagenet
CUDA_VISIBLE_DEVICES=3 python main_imagenet.py --dist-url tcp://localhost:10009 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset imagenet --arch resnet18 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/datasets/ImageNet32/Imagenet32x32 
--exp_dir ./experiment/imagenet 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos 
| tee -a zz_output_imagenet.log 


# for aircraft
CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url tcp://localhost:10008 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset aircraft --arch resnet50 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/datasets/fgvc-aircraft-2013b 
--exp_dir ./experiment/aircraft 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos  
| tee -a zz_aircraft.log 


# for cars196
CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url tcp://localhost:10001 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset cars196 --arch resnet50 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/datasets/cars196
--exp_dir ./experiment/cars196 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos 
| tee -a zz_cars196.log 

# for flowers102
CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url tcp://localhost:10001 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset flowers102 --arch resnet50 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/ssd2T/Datasets/flowers102 
--exp_dir ./experiment/flowers102 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos 
| tee -a zz_flowers102.log

# for nabirds
CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url tcp://localhost:10002 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset nabirds --arch resnet50 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 
--data /media/disk12T/2022-sgh/datasets/NABirds 
--warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos --exp_dir ./experiment/nabirds_05 
| tee -a zz_nabirds_05.log

# for cub200
CUDA_VISIBLE_DEVICES=0,1 python main.py --dist-url tcp://localhost:10010 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset cub200 --arch resnet50 
--data /media/ssd2T/Datasets/cub200/CUB_200_2011/CUB_200_2011 
--exp_dir ./experiment/cub200 
--img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 --warmup_epoch 10 --epochs 100 --workers 8 --mlp --aug-plus --cos 
| tee -a zz_cub200.log


# ==============================================================================
# for iNaturalist_2019
# kingdom phylum class order family genus species
CUDA_VISIBLE_DEVICES=2 python main_iNaturalist.py --dist-url tcp://localhost:10009 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset iNaturalist2019 --arch resnet18 --hie_level phylum --img_size 224 --lr 0.03 --batch-size 128 --moco-k 65536 --moco-t 0.2 
--data /media/ssd2T/Datasets/iNaturalist2019 
--exp_dir ./experiment/iNaturalist2019_phylum 
--warmup_epoch 10 --epochs 100 --workers 16 --mlp --aug-plus --cos 
| tee -a zz_iNaturalist2019_phylum.log 
# ==============================================================================




# ==============================================================================
# for eval to calculate the metrics
CUDA_VISIBLE_DEVICES=2,3 python main.py --dist-url tcp://localhost:10009 --multiprocessing-distributed --world-size 1 --rank 0 
--dataset cifar100 --arch resnet50 --img_size 224 --lr 0.03 --batch-size 256 --moco-k 65536 --moco-t 0.2 --workers 4 
--data /media/disk12T/2022-sgh/lxy/data 
--resume experiment/cifar100_10times/checkpoint_0099.pth.tar 
--eval 
# ==============================================================================
