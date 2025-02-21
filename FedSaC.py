import copy
import math
import random
import time
from FedSaC.test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from FedSaC.config import get_args
from FedSaC.utils import aggregation_by_graph, update_graph_matrix_neighbor
from FedSaC.model import simplecnn, textcnn
from FedSaC.prepare_data import get_dataloader
from FedSaC.attack import *
from sklearn.decomposition import PCA
import glog as log
import sys
import os
import json

def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl,
                          data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list, pca):
    """
    在每个客户端上进行本地训练
    :param args: 配置参数的对象
    :param round: 通信轮数
    :param nets_this_round: 一个字典，键是客户端的 ID，值是对应的模型对象，表示当前轮次中参与训练的客户端模型。
    :param cluster_models: 一个字典，键是客户端的 ID，值是对应的聚类模型向量，用于正则化训练过程。
    :param train_local_dls: 一个列表，包含每个客户端的本地训练数据加载器
    :param val_local_dls: 一个列表，包含每个客户端的本地验证数据加载器
    :param test_dl:  一个数据加载器，包含整个测试集，用于评估模型的泛化能力
    :param data_distributions: 一个二维数组，记录每个客户端的每个类别的样本比例
    :param best_val_acc_list:  一个列表，记录每个客户端的最佳验证准确率
    :param best_test_acc_list: 一个列表，记录每个客户端的最佳测试准确率
    :param benign_client_list: 一个列表，包含所有良性客户端的 ID，用于区分正常客户端和可能的恶意客户端
    :param pca: 一个 PCA 对象，用于特征降维
    :return:
    principal_list: 一个列表，包含每个客户端的主成分分析（PCA）结果，即正交基。
    mean_personalized_acc: 良性客户端的平均个性化测试准确率。

'''
This is local train by FedSaC
'''
    x (128,3,32,32) target (128)
    """
    principal_list = []  # 存储每个客户端的主成分分析（PCA）结果，即正交基
    for net_id, net in nets_this_round.items():

        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])  # 验证集上的准确率
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl,
                                                                                      data_distribution)  # 测试集上的准确率

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            log.info(
                '>> Round {} | Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(
                    round, net_id, personalized_test_acc, generalized_test_acc))
        else:
            log.info('Round {} | Client {} not in benign_client_list'.format(round, net_id))
        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                                   weight_decay=args.reg,
                                   amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                  weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        if net_id in cluster_models:
            cluster_model = cluster_models[net_id].cuda()

        # net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            # x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            if net_id in cluster_models:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(
                    flatten_model)  # 额外的正则化损失
                loss2.backward()

            loss.backward()
            optimizer.step()

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)
            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            log.info(
                '>> Round {} | Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(
                    round, net_id, personalized_test_acc, generalized_test_acc))
        # 对良性客户端的模型进行特征提取，并通过PCA（主成分分析）降维，提取出正交基（orthogonal basis），并将这些正交基存储到principal_list中
        log.info('>> concatenate')
        if net_id in benign_client_list:
            net.eval()
            # net.cuda()
            feature_list = []  # 存储从每个数据批次中提取的特征
            with torch.no_grad():
                for x, _ in train_local_dl:
                    # x = x.cuda()
                    feature = net.extract_feature(x)  # 提取输入数据的特征 (128,84)
                    feature_list.append(feature.cpu().numpy())
            feature_array = np.concatenate(feature_list, axis=0)  # (13333,84)
            pca.fit_transform(feature_array)  # 将特征数组降维到指定的维度
            orthogonal_basis = pca.components_  # 获取PCA对象pca的主成分，即正交基 (3,84)
            principal_list.append(orthogonal_basis)  # 存储所有良性客户端的正交基
        log.info('>> concatenate end')
        net.to('cpu')
    return principal_list, np.array(best_test_acc_list)[np.array(benign_client_list)].mean()


args, cfg = get_args()
if args.partition == "noniid_1":
    args.partition = "noniid"
    args.beta = 0.1
elif args.partition == "noniid_2":
    args.partition = "noniid"
    args.beta = 0.5

seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

k_principal = int(args.k_principal)
pca = PCA(n_components=k_principal)  # 通过 PCA 将数据降维到 3 维，提取了数据中的关键特征，减少了数据的维度，

n_party_per_round = 3  # 每轮客户端数量为10
party_list = [i for i in range(args.n_parties)]  # 客户端索引
party_list_rounds = []  # 每一轮通信中参与的客户端索引列表
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_parties * (1 - args.attack_ratio)))
benign_client_list.sort()
log.info('>> -------- Benign clients: {} --------'.format(benign_client_list))

'''
train_local_dls 一个列表，包含每个客户端的训练数据加载器
val_local_dls 一个列表，包含每个客户端的验证数据加载器
test_dl  一个数据加载器，包含整个测试集
net_dataidx_map 一个字典，记录每个客户端分配到的数据索引
traindata_cls_counts 一个二维数组，记录每个客户端的每个类别的样本数量
data_distributions 一个二维数组，记录每个客户端的每个类别的样本比例
'''

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(
    args)

if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn

hidden_dim = args.hidden_dim  # 84
global_model = model(hidden_dim, cfg['classes_size'])  # simpleCNN model
global_parameters = global_model.state_dict()  # dict {'base.conv1.weight': tensor, 'base.conv1.bias': tensor, ...}
local_models = []
best_val_acc_list, best_test_acc_list = [], []
dw = []
for i in range(cfg['client_num']):
    local_models.append(model(hidden_dim, cfg['classes_size']))
    dw.append({key: torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models) - 1)  # Collaboration Graph
graph_matrix[range(len(local_models)), range(len(local_models))] = 0  # 对角线元素为0

for net in local_models:
    net.load_state_dict(global_parameters)

cluster_model_vectors = {}  # 存储每个客户端（或参与方）的聚类模型向量
total_round = cfg["comm_round"]
for round in range(total_round):
    log.info('>> -------- Round %d --------', round)
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        log.info('>> Clients in this round : %d', party_list_this_round)
    else:
        print(f'>> ALl Clients in this round : {party_list_this_round}')
    nets_this_round = {k: local_models[k] for k in party_list_this_round}  # 本轮通信中参与的客户端的模型 字典 {0: model, 1: model, ...}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
    principal_list, mean_personalized_acc = local_train_pfedgraph(args, round, nets_this_round, cluster_model_vectors,
                                                                  train_local_dls, val_local_dls,
                                                                  test_dl, data_distributions, best_val_acc_list,
                                                                  best_test_acc_list, benign_client_list, pca)

    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])  # 总数据量
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in
                     party_list_this_round}  # 每个参与训练的客户端的数据量占比，并存储在一个字典中。

    manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)
    if round / total_round > args.alpha_bound:  # args.alpha_bound 1
        matrix_alpha = 0
    else:
        matrix_alpha = args.matrix_alpha  # 0.9
    graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters, principal_list, dw,
                                                fed_avg_freqs, matrix_alpha,
                                                args.matrix_beta, args.complementary_metric,
                                                args.difference_measure)  # Graph Matrix is not normalized yet
    cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round,
                                                 global_parameters)  # Aggregation weight is normalized here

    log.info('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    with open(args.exp_dir, 'a') as f:
        f.write('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-' * 80)
open(os.path.join(args.exp_dir, 'done'), 'a').close()
