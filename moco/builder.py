# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
import numpy as np
torch.autograd.set_detect_anomaly(True)

#maskcon model引入方式
from torchvision.models import resnet



class ModelBase(nn.Module):
    """
    For small size figures:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, base_encoder, args, dim=128):
        super(ModelBase, self).__init__()

        self.net = base_encoder(pretrained=True)
        # for resnet18
        if args.img_size <=64:
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity()
        self.dim_mlp = self.net.fc.weight.shape[1]
        self.net.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(self.dim_mlp, self.dim_mlp),
            nn.ReLU(),
            nn.Linear(self.dim_mlp, dim)
        )

        self.classifer = nn.Linear(self.dim_mlp, args.category_num)

    def forward(self, x, feat=False):
        x = self.net(x)
        if feat:
            return x
        else:
            proj = self.projector(x)
            cls = self.classifer(x)
            return x, proj, cls


    

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

# 修改内容：(新函数都加在forward上面)
# 1.K的大小
# 2.instance_select
# 3.is_eval->compute feature
# 4.加入聚类算法-> get_protos
    def __init__(self, base_encoder, args, dim=128, K=5120, m=0.999, T=0.2):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.instance_selection = True

        


        self.encoder_q = ModelBase(base_encoder, args, dim)
        self.encoder_k = ModelBase(base_encoder, args, dim)

        self.register_buffer("embedding_dim", torch.zeros(1, dtype=torch.long))
        self.embedding_dim[0] = self.encoder_q.dim_mlp

        # self.embedding_dim = nn.Parameter(torch.Tensor(self.encoder_q.dim_mlp))
        self.mlp_dim = dim



        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_index", torch.arange(0, K))

        # create the labels queue
        self.register_buffer("label_queue", torch.zeros([K,1]))
        self.register_buffer("label_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue_index", torch.arange(0, K))#不一定有用

        self.buffer_dict = dict()
        self.mined_index = list()        

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, index=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        if index is not None:
            index = concat_all_gather(index)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        if index is not None:
            self.queue_index[ptr: ptr + batch_size] = index
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def _label_dequeue_and_enqueue(self, labels, index=None):
        # gather labels before updating queue
        labels = concat_all_gather(labels)
        if index is not None:
            index = concat_all_gather(index)
        batch_size = labels.shape[0]

        ptr = int(self.label_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the labels at ptr (dequeue and enqueue)
        self.label_queue[ptr : ptr + batch_size, 0] = labels
        if index is not None:
            self.label_queue_index[ptr: ptr + batch_size] = index
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.label_queue_ptr[0] = ptr       

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    

    @torch.no_grad()
    def sample_neg_instance(self, im2cluster, centroids, density, index, labels_queue, epoch):
        """
        mining based on the clustering results
        """
        queue_p_samples = []
        # print('len:im2cluster', len(im2cluster))
        for layer in range(len(im2cluster)):
            proto_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids[layer].permute(1, 0)).div(0.1)  
            density[layer] = density[layer].clamp(min=1e-3)
            proto_logit /= density[layer] #[K,80]
            label = im2cluster[layer][index] #[batch_size]
            logit = proto_logit.clone().detach().softmax(-1)
            p_sample = 1 - logit[:, label].t() #[batch_size,K]

            
            #加入第二层mask(label mask保证他在类内选)
            second_mask = (self.label_queue.clone().detach() == labels_queue[:]).float().t()
            p_sample = p_sample * second_mask
            p_sample = p_sample / p_sample.max(dim=1, keepdims=True)[0]

            new_mask = 1 - second_mask
            
            queue_p_samples.append(p_sample)
        

        self.selected_masks = []
        avg_sample_ratios = []
        for p_sample in queue_p_samples:
            # print('p_sample:{}'.format(p_sample))
            neg_sampler = torch.distributions.bernoulli.Bernoulli(p_sample.clamp(0.0, 0.999))
            selected_mask = neg_sampler.sample() # [N_q, N_queue]
            # print('selected_mask:{}'.format(selected_mask))
            # print('selected_mask non_zero:{}'.format((selected_mask != 0).sum(dim = 1)))
            # different = after_mask - (selected_mask != 0).sum(dim = 1)
            # print(different)
            
            try:
                self.selected_masks.append(selected_mask + new_mask)
                avg_sample_ratios.append(p_sample.mean())
            except:
                # when no samples are selected
                selected_mask = torch.ones([index.shape[0], self.queue.shape[1]]).cuda()
                self.selected_masks.append(selected_mask)
        return self.selected_masks, avg_sample_ratios   


    def extract_mlp(self, im_k):
        _, k, _ = self.encoder_k(im_k) 
        k = nn.functional.normalize(k, dim=1)            
        return k

    def extract_embedding_k(self, im_k):
        embedding_k, _, _ = self.encoder_k(im_k) 
        # embedding_k = nn.functional.normalize(embedding_k, dim=1)            
        return embedding_k

    def extract_embedding_q(self, im_q):
        embedding, _, _ = self.encoder_q(im_q) 
        # q_before = nn.functional.normalize(q_before, dim=1)            
        return embedding
        
    
    def get_protos(self, q, index, cluster_result):
        # prototypical contrast
        if cluster_result is not None:  
            proto_selecteds = []
            temp_protos = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                proto_selecteds.append(prototypes)
                temp_protos.append(density)
            return proto_selecteds, temp_protos
        else:
            return None, None

    def forward(self, im_q, im_k, is_mlp_k=False, is_embedding_q=False, is_embedding_k=False, cluster_result=None, index=None, labels_queue=None, epoch=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_embedding, q, q_ce_logit = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        if is_mlp_k:
            return self.extract_mlp(im_k)
        if is_embedding_q:
            return self.extract_embedding_q(im_q)
        if is_embedding_k:
            return self.extract_embedding_k(im_k)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            _, k, _ = self.encoder_k(im_k)  # keys: NxC
            # k_before, k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        #加入聚类以及更改负样本选择方式
        proto_selected, temp_protos = self.get_protos(q, index, cluster_result)

        # negative logits: NxK
        #print('self_instance_selection', self.instance_selection)
        final_pos_weight = None
        if self.instance_selection and cluster_result is not None:
            try:
                # print('yes try selection')
                pos_selection = 1
                #加入pos挑选模块
                # final_pos_weight = self.sample_pos_instance(q, labels_queue)
                if pos_selection:
                    # final_pos_weight = self.sample_pos_instance(q, labels_queue)
                    pos_im2cluster = cluster_result['im2cluster']
                    centroids = proto_selected
                    density = temp_protos
                    # real_int_pos_logit = torch.mm(q, self.queue.clone().detach())
                    real_int_pos_logit = torch.mm(k, self.queue.clone().detach())
                    pos_logit = torch.mm(self.queue.clone().detach().permute(1, 0), centroids[0].permute(1, 0)).div(0.1) 
                    pos_density = density[0].clamp(min=1e-3)
                    pos_logit = pos_logit / pos_density
                    pos_label = pos_im2cluster[0][index]
                    pos_logit = pos_logit.clone().detach().softmax(-1)
                    pos_sample = pos_logit[:, pos_label].t()
                    second_mask = (self.label_queue.clone().detach() == labels_queue[:]).float().t()
                    pos_sample = pos_sample * second_mask

                    pos_sample = pos_sample / pos_sample.max(dim=1, keepdims=True)[0]

                    pos_sampler = torch.distributions.bernoulli.Bernoulli(pos_sample.clamp(0.0, 0.999))
                    pos_sample_mask = pos_sampler.sample()
                    # print('mask后的正样本挑选个数：', (pos_sample_mask == 1).sum(dim=1))
                    ########################################################################################################
                    #下一步要进行加权操作
                    #异常排除，即选出的正样本数量为0
                    #1.[1//1]加权方式(挑选部分的权值其实大于1)
                    masked_pos_sim = (real_int_pos_logit / 0.1) * pos_sample_mask
                    # tmp_check = masked_pos_sim
                    masked_pos_sim = masked_pos_sim - masked_pos_sim.max(dim=1, keepdim=True)[0]
                    pos_weight = masked_pos_sim.exp() * pos_sample_mask
                    pos_weight_sum = torch.sum(pos_weight, dim=1, keepdim=True) + 1e-10
                    tmp = pos_weight / pos_weight_sum
                    tmp = tmp / (tmp.max(dim=1, keepdim=True)[0] + 1e-10)
                    final_pos_weight = torch.zeros(len(q), self.K + 1).cuda()
                    final_pos_weight[:, 0] = 1
                    final_pos_weight[:, 1:] = tmp
                    final_pos_weight = final_pos_weight / final_pos_weight.sum(dim=1, keepdim=True)
                    final_pos_weight = final_pos_weight.clone().detach()                                        
                
                #负样本挑选模块
                self.selected_masks, sample_ratios = self.sample_neg_instance(cluster_result['im2cluster'], proto_selected, temp_protos, index, labels_queue=labels_queue, epoch=epoch)       
                self.buffer_dict['avg_sample_ratios'] = sum(sample_ratios) / len(sample_ratios)
                l_neg = list()
                for selected_mask in self.selected_masks:
                    logit = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
                    mask = selected_mask.clone().float()
                    # print('mask后的负样本挑选个数：', (mask == 1).sum(dim=1))
                    # fake = mask + final_pos_weight[:, 1:] #有问题，应该加的是pos_mask而不是后面的权重 
                    #对应topk操作
                    # mask = mask + pos_mask
                    #对应所有正样本操作
                    mask = mask + pos_sample_mask
                    # print('true mask', torch.sum(mask, dim=1))
                    mask = mask.detach()
                    # print('final negmask sum:', torch.sum(mask, dim=1))

                    #注释掉下面一行,即不mask东西,即logit就是logit,负样本全用上,只是正样本权重不一样而已
                    l_neg.append(logit * mask)
                    # l_neg.append(logit)
                    #print('len l_neg', len(l_neg))

                # print('yes neg selection')

            except Exception as e:
                print(e)
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        if isinstance(l_neg, list):
            # print('lpos size',l_pos_list[1].size())
            # print('lneg0 size',l_neg[0].size())
            logits = [torch.cat([l_pos, l_n], dim=1)/self.T for l_n in l_neg]
            
            #打印前三行非0元素以及其索引
            # rows = logits[0][:3]
            # # 找出非零元素的索引
            # nonzero_indices = torch.nonzero(rows)
            # 打印非零元素的值和索引
            # nonzero_values = rows[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            # nonzero_indices = nonzero_indices.tolist()
            # for value, idx in zip(nonzero_values, nonzero_indices):
            #     row, col = idx
            #     print(f"logits: {value}, Index: ({row}, {col})")
            
            # logits = [torch.cat([l_p, l_neg[0]], dim=1)/self.T for l_p in l_pos_list] #应该是4个logit才对
            # print('len(logits):', len(logits))
            labels = [torch.zeros(logit.shape[0], dtype=torch.long).cuda() for logit in logits]
            # print('len(labels):', len(labels)) #这里也是4个才对，并且都是0向量
        else:
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.T
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, index)
        self._label_dequeue_and_enqueue(labels_queue, index)
        if final_pos_weight is None:
            return logits, labels, q_ce_logit, None
        else:
            return logits, labels, q_ce_logit, final_pos_weight



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
