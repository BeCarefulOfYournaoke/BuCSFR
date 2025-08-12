import numpy as np
import faiss
import numba as nb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

import time


import networkx as nx
import numpy as np


def run_hkmeans(x, args, fea_dim=128):
    name_list = args.name_list
    """
    This function is a hierarchical 
    k-means: the centroids of current hierarchy is used
    to perform k-means in next stepW
    x is sample feature/
    """
    data_counter = 0
    emb_lenth = fea_dim
    embedding_labels = x
    x = x[:, 0:emb_lenth]
    print("len of dataset is: ", np.shape(x)[0])
    print("performing kmeans clustering")
    results = {"im2cluster": [], "centroids": [], "density": [], "cluster_num_perc": []}
    print("now we do the cluster")
    print("before merge:total is {}, different is {}".format(sum(args.cluster_num_perc), args.cluster_num_perc))

    # intialize faiss clustering parameters
    category_num = args.category_num
    print("category number:{}".format(category_num))

    final_im2cluster = [-1 for _ in range(np.shape(x)[0])]
    for i in range(category_num):  # category_num
        temp_index = embedding_labels[:, emb_lenth].astype(np.int8) == i
        per_x = x[temp_index]
        print("num of samples in {}-th class is: {}".format(i, per_x.shape))

        d = per_x.shape[1]
        k = int(args.cluster_num_perc[i])
        if k > per_x.shape[0]:
            args.cluster_num_perc[i] = per_x.shape[0]
            k = per_x.shape[0]
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = 0
        if args.dataset == "cars196" or args.dataset == "aircraft" or args.dataset == "flowers102"or args.dataset == "nabirds" or args.dataset == "cub200":
            clus.min_points_per_centroid = 4
            clus.max_points_per_centroid = 200
        elif args.dataset == "imagenet":
            clus.min_points_per_centroid = 10
            clus.max_points_per_centroid = 1000
        else:
            clus.max_points_per_centroid = 1710
            clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        res.setTempMemory(1 * 1024 * 1024 * 1024)
        
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(per_x, index)
        D, I = index.search(per_x, 1)  # for each sample, find cluster distance and assignments

        im2cluster = [int(n[0]) for n in I]

        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        if len(set(im2cluster)) == 1:
            print("Warning! All samples are assigned to one cluster")

        temp_im2cluster = im2cluster[:]

        # start merging clusters
        norm = np.linalg.norm(centroids, axis=1).reshape(np.shape(centroids)[0], 1)
        norm_centroids = centroids / norm

        centroids_sim_matrix = np.ones((np.shape(norm_centroids)[0], np.shape(norm_centroids)[0]))

        # entropy matrix
        a = np.ones_like(centroids_sim_matrix) * -1
        a = new_new_entropy_counter(a, per_x, im2cluster)

        rows, cols = np.triu_indices_from(a, k=1)
        sorted_indices = np.argsort(a[rows, cols])
        sorted_elements = a[rows, cols][sorted_indices]
        sorted_rows = rows[sorted_indices]
        sorted_cols = cols[sorted_indices]
        new_centroids = np.zeros((1, emb_lenth))
        new_flag = 0
        merge_flag = [0 for j in range(args.cluster_num_perc[i])]
        im2cluster = np.array(im2cluster)
        temp_im2cluster = np.array(temp_im2cluster)
        for element, row, col in zip(sorted_elements, sorted_rows, sorted_cols):
            if merge_flag[row] == 1 or merge_flag[col] == 1:
                continue
            elif element <= 0:
                continue
            else:
                centroid_j = norm_centroids[row, :].reshape(1, emb_lenth)
                centroid_k = norm_centroids[col, :].reshape(1, emb_lenth)
                j_data = per_x[np.array(im2cluster) == row, :]
                k_data = per_x[np.array(im2cluster) == col, :]
                combined_data = np.concatenate((j_data, k_data), axis=0)

                thres = a[row][col]

                check = np.mean(combined_data, axis=0).reshape(1, emb_lenth)
                check = check / np.linalg.norm(check, axis=1)

                if 0 < thres and thres < args.threshold:
                    print("merge:category {}, new_thres_calculate is:{}, number:{}".format(name_list[i], thres, args.cluster_num_perc[i] - 1))
                    new_clus_centroids = check
                    if new_flag == 0:
                        new_centroids[0, :] = new_clus_centroids
                        new_flag = 1
                    else:
                        temp = np.zeros([1, emb_lenth])
                        temp[0, :] = new_clus_centroids
                        new_centroids = np.concatenate((new_centroids, temp), axis=0)
                    merge_flag[col] = 1
                    merge_flag[row] = 1

                    # update im2cluster
                    for idx in range(len(im2cluster)):
                        if im2cluster[idx] == row or im2cluster[idx] == col:
                            temp_im2cluster[idx] = np.shape(new_centroids)[0] - 1
                        else:
                            continue
                    break
                else:
                    # do not merge
                    print("not working,category {}:, thres_calculate: {}, number:{}".format(name_list[i], thres, args.cluster_num_perc[i]))
                    break

        for j in range(args.cluster_num_perc[i]):
            if merge_flag[j] == 1:
                continue
            else:
                if merge_flag[j] == 0 and new_flag == 0:
                    new_centroids[0, :] = centroids[j]
                    new_flag = 1
                    merge_flag[j] = 1
                    for idx in range(len(temp_im2cluster)):
                        if temp_im2cluster[idx] == j:
                            temp_im2cluster[idx] = 0
                elif merge_flag[j] == 0 and new_flag == 1:
                    temp = np.zeros([1, emb_lenth])
                    temp[0, :] = centroids[j]
                    new_centroids = np.concatenate((new_centroids, temp), axis=0)
                    merge_flag[j] = 1
                    for idx in range(len(temp_im2cluster)):
                        if im2cluster[idx] == j:
                            temp_im2cluster[idx] = np.shape(new_centroids)[0] - 1
        im2cluster = im2cluster.tolist()
        temp_im2cluster = temp_im2cluster.tolist()
        if i == 0:
            center_index = 0
        else:
            center_index = sum(args.cluster_num_perc[0:i])
        temp_im2cluster = [int(n + center_index) for n in temp_im2cluster]
        if i == 0:
            final_centroids = new_centroids
            final_im2cluster = np.array(final_im2cluster)
            final_im2cluster[temp_index] = temp_im2cluster
            final_im2cluster = final_im2cluster.tolist()
        else:
            final_centroids = np.concatenate((final_centroids, new_centroids), axis=0)
            final_im2cluster = np.array(final_im2cluster)
            final_im2cluster[temp_index] = temp_im2cluster
            final_im2cluster = final_im2cluster.tolist()
        data_counter += temp_index.sum()
        args.cluster_num_perc[i] = np.shape(new_centroids)[0]

        res.noTempMemory()

    print("merging num: ", data_counter)
    print("data len: ", np.shape(x)[0])

    final_density = np.ones(int(sum(args.cluster_num_perc))) * 0.1
    final_centroids = torch.Tensor(final_centroids).cuda()
    final_centroids = nn.functional.normalize(final_centroids, p=2, dim=1)
    final_density = torch.Tensor(final_density).cuda()
    final_im2cluster = torch.LongTensor(final_im2cluster).cuda()
    tensor_cluster_num_perc = torch.tensor(args.cluster_num_perc).cuda()
    results["centroids"].append(final_centroids)
    results["im2cluster"].append(final_im2cluster)
    results["density"].append(final_density)
    results["cluster_num_perc"].append(tensor_cluster_num_perc)

    args.num_cluster = sum(args.cluster_num_perc)
    print("after merge, num of cluster is:{}".format(args.num_cluster))
    print("different is:{}".format(args.cluster_num_perc))
    for i in range(category_num):
        print("category {}: num: {} ".format(name_list[i], args.cluster_num_perc[i]))
    return results


@nb.njit(parallel=True)
def entropy_counter(z, lamda, order=1):
    norm = np.sqrt(np.sum(z**2, axis=1))
    z /= norm[:, np.newaxis]
    if np.shape(z)[0] <= np.shape(z)[1]:
        c = z @ z.T
        c = c * lamda
    else:
        c = z.T @ z
        c = c * lamda

    power_matrix = c
    sum_matrix = np.zeros_like(power_matrix)

    for k in range(1, order + 1):
        if k > 1:
            power_matrix = np.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else:
            sum_matrix -= power_matrix / k

    trace = np.trace(sum_matrix)

    return trace


def new_entropy_counter(z, lamda):

    norm = np.linalg.norm(z, axis=1).reshape(np.shape(z)[0], 1)
    z /= norm
    N, D = z.shape
    if np.shape(z)[0] <= D:
        # c = z @ z.T
        c = np.dot(z, z.T)
        c = c * lamda
        I = np.eye(np.shape(c)[0])
    else:
        c = np.dot(z.T, z)
        c = c * lamda
        I = np.eye(np.shape(c)[0])
    power_matrix = c + I
    determinant = np.linalg.det(power_matrix)
    log_deter = np.log2(determinant)
    return log_deter


@nb.jit(nopython=True)
def new_new_entropy_counter(a, per_x, im2cluster):
    exiu = 16
    b = np.array(im2cluster)
    for new_j in range(np.shape(a)[0]):
        if new_j == np.shape(a)[0] - 1:
            break
        for new_k in range(new_j + 1, np.shape(a)[0]):
            j_data = per_x[b == new_j, :]
            k_data = per_x[b == new_k, :]
            counter_j = 0
            counter_k = 0
            for i in range(len(im2cluster)):
                if im2cluster[i] == new_j:
                    counter_j += 1
                if im2cluster[i] == new_k:
                    counter_k += 1

            counter_combine = counter_j + counter_k

            lamda_j = 1 / (counter_j * (exiu / 128))
            lamda_k = 1 / (counter_k * (exiu / 128))
            lamda_combine = 1 / (counter_combine * (exiu / 128))

            combined_data = np.concatenate((j_data, k_data), axis=0)

            N, D = j_data.shape
            if np.shape(j_data)[0] <= D:
                # print('NxN')
                c = np.dot(j_data, j_data.T)
                c = c * lamda_j
                I = np.eye(np.shape(c)[0])
            else:
                c = np.dot(j_data.T, j_data)
                c = c * lamda_j
                I = np.eye(np.shape(c)[0])
            power_matrix = c + I
            determinant = np.linalg.det(power_matrix)
            log_deter = np.log2(determinant)
            L_j = log_deter * ((counter_j + 128) / 2)

            # k data entropy
            N, D = k_data.shape
            if np.shape(k_data)[0] <= D:
                c = np.dot(k_data, k_data.T)
                c = c * lamda_k
                I = np.eye(np.shape(c)[0])
            else:
                c = np.dot(k_data.T, k_data)
                c = c * lamda_k
                I = np.eye(np.shape(c)[0])
            power_matrix = c + I
            determinant = np.linalg.det(power_matrix)
            log_deter = np.log2(determinant)
            L_k = log_deter * ((counter_k + 128) / 2)

            # combine data entropy
            N, D = combined_data.shape
            if np.shape(combined_data)[0] <= D:
                c = np.dot(combined_data, combined_data.T)
                c = c * lamda_combine
                I = np.eye(np.shape(c)[0])
            else:
                c = np.dot(combined_data.T, combined_data)
                c = c * lamda_combine
                I = np.eye(np.shape(c)[0])
            power_matrix = c + I
            determinant = np.linalg.det(power_matrix)
            log_deter = np.log2(determinant)
            L_combine = log_deter * ((counter_combine + 128) / 2)

            I_jk = L_j + L_k - L_combine

            if counter_j >= counter_k:
                thres = L_k / I_jk
            else:
                thres = L_j / I_jk

            a[new_j, new_k] = thres
    return a


@nb.jit(nopython=True)
def distance_count(test_features, train_features):
    num_test = test_features.shape[0]
    num_train = train_features.shape[0]
    num_features = test_features.shape[1]
    distances = np.zeros((num_test, num_train))

    for i in range(num_test):
        test_sample = test_features[i]

        # Compute distances between test_sample and all train_samples
        for j in range(num_train):
            distance = 0.0
            for d in range(num_features):
                diff = train_features[j, d] - test_sample[d]
                distance += diff * diff
            distances[i, j] = distance

    return distances


@nb.jit(nopython=True)
def compute_top1_accuracy(predicted_labels, test_labels, k):
    num_test = test_labels.shape[0]
    top1_correct = 0
    predicted_labels = predicted_labels[:, 0:k]

    for i in range(num_test):
        unique_labels, label_counts = np.unique(predicted_labels[i], return_counts=True)
        predict = unique_labels[np.argmax(label_counts)]
        if test_labels[i] == predict:
            top1_correct += 1

    top1_accuracy = top1_correct / num_test
    return top1_accuracy


def compute_features(eval_loader, model, args, is_mlp_k=True, is_embedding_q=False, is_embedding_k=False, embedding_dim=2048):
    print("Computing features...")
    model.eval()

    if is_mlp_k:
        dim = args.dim
    else:
        dim = embedding_dim

    features = torch.zeros(len(eval_loader.dataset), dim).cuda()
    targets = torch.zeros(len(eval_loader.dataset), 1).cuda()
    fine_targets = torch.zeros(len(eval_loader.dataset), 1).cuda()
    # for i, (images, index, target) in enumerate(tqdm(eval_loader)):
    eval_bar = tqdm(eval_loader)

    for i, (images, index, target, fine_target) in enumerate((eval_bar)):
        with torch.no_grad():
            target = target.cuda(non_blocking=True)
            fine_target = fine_target.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            feat = model(im_q=images, im_k=images, is_mlp_k=is_mlp_k, is_embedding_q=is_embedding_q, is_embedding_k=is_embedding_k)
            features[index] = feat
            targets[index] = target.unsqueeze(-1).float()
            fine_targets[index] = fine_target.unsqueeze(-1).float()

            eval_bar.set_description("Train mlp inference:")

    features = torch.cat((features, targets), axis=1)
    features = torch.cat((features, fine_targets), axis=1)
    return features


def retrieval(model, val_loader, eval_loader, K, cluster_result, args, embedding_dim=2048):
    chunks = 500
    print("Computing train features...(for knn)")
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), embedding_dim).cuda()
    targets = torch.zeros(len(eval_loader.dataset), 1).cuda()
    fine_targets = torch.zeros(len(eval_loader.dataset), 1).cuda()
    eval_bar = tqdm(eval_loader)
    for i, (images, index, target, fine_target) in enumerate(eval_bar):
        with torch.no_grad():
            target = target.cuda(non_blocking=True)
            fine_target = fine_target.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            feat = model(im_q=images, im_k=images, is_embedding_q=True)
            features[index] = feat
            targets[index] = target.unsqueeze(-1).float()
            fine_targets[index] = fine_target.unsqueeze(-1).float()

            eval_bar.set_description("Train embedding inference:")

    train_features = torch.cat((features, targets), axis=1)
    train_features = torch.cat((train_features, fine_targets), axis=1)
    print("train_feature size:", train_features.size())

    print("Computing test features...(for recall)")
    features = torch.zeros(len(val_loader.dataset), embedding_dim).cuda()
    targets = torch.zeros(len(val_loader.dataset), 1).cuda()
    fine_targets = torch.zeros(len(val_loader.dataset), 1).cuda()
    test_bar = tqdm(val_loader)
    for i, (images, index, target, fine_target) in enumerate(test_bar):
        with torch.no_grad():
            target = target.cuda(non_blocking=True)
            fine_target = fine_target.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            feat = model(im_q=images, im_k=images, is_embedding_q=True)
            features[index] = feat
            targets[index] = target.unsqueeze(-1).float()
            fine_targets[index] = fine_target.unsqueeze(-1).float()

            test_bar.set_description("Test embedding inference:")

    test_features = torch.cat((features, targets), axis=1)
    test_features = torch.cat((test_features, fine_targets), axis=1)
    print("test_feature size:", test_features.size())

    # recall calculation
    test_feat_norm = nn.functional.normalize(test_features[:, 0:embedding_dim], dim=1)
    test_label = test_features[:, -1].unsqueeze(-1)
    split = torch.tensor(np.linspace(0, len(test_feat_norm), chunks + 1, dtype=int), dtype=torch.long).to(test_features.device)

    recall = [[] for i in K]
    ids = [torch.tensor([]).to(test_features.device) for i in K]
    correct = [torch.tensor([]).to(test_features.device) for i in K]
    k_max = np.max(K)

    with torch.no_grad():
        for j in range(chunks):
            torch.cuda.empty_cache()
            part_feature = test_feat_norm[split[j] : split[j + 1]]
            similarity = torch.einsum("ab,bc->ac", part_feature, test_feat_norm.T)

            topmax = similarity.topk(k_max + 1)[1][:, 1:]
            del similarity
            retrievalmax = test_label[topmax].squeeze()
            for k, i in enumerate(K):
                anchor_label = test_label[split[j] : split[j + 1]].repeat(1, i)
                topi = topmax[:, :i]
                retrieval_label = retrievalmax[:, :i]
                correct_i = torch.sum(anchor_label == retrieval_label, dim=1, keepdim=True)
                correct[k] = torch.cat([correct[k], correct_i], dim=0)
                ids[k] = torch.cat([ids[k], topi], dim=0)

        # calculate recall @ K
        num_sample = len(test_feat_norm)
        for k, i in enumerate(K):
            acc_k = float((correct[k] > 0).int().sum() / num_sample)
            recall[k] = acc_k


    # knn calculation
    knn = [[] for i in K]
    train_feat_norm = nn.functional.normalize(train_features[:, 0:embedding_dim], dim=1)
    train_label = train_features[:, -1].unsqueeze(-1)
    correct = [torch.tensor([0]).to(test_features.device) for i in K]
    split = torch.tensor(np.linspace(0, len(test_feat_norm), chunks + 1, dtype=int), dtype=torch.long).to(test_features.device)

    # test debug
    with torch.no_grad():
        for j in range(chunks):
            torch.cuda.empty_cache()
            part_feature = test_feat_norm[split[j] : split[j + 1]]
            similarity = torch.einsum("ab,bc->ac", part_feature, train_feat_norm.T)
            topmax = similarity.topk(k_max)[1][:, 0:]
            del similarity

            retrievalmax = train_label[topmax].squeeze()
            for k, i in enumerate(K):
                anchor_label = test_label[split[j] : split[j + 1]]
                retrieval_label = retrievalmax[:, :i]
                top_retrieval_label, _ = torch.mode(retrieval_label, dim=1, keepdim=True)
                correct_i = torch.sum(anchor_label.squeeze() == top_retrieval_label.squeeze())
                correct[k] += correct_i
        # calculate recall @ K
        num_sample = len(test_feat_norm)
        for k, i in enumerate(K):
            acc_k = float(correct[k] / num_sample)
            knn[k] = acc_k
            
    return recall, knn
