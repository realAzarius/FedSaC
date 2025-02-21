import torch
import numpy as np
import copy
import cvxpy as cp
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import glog as log


def compute_local_test_accuracy(model, dataloader, data_distribution):
    model.eval()

    toatl_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution).sum()

    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    """
    计算模型差异矩阵
    :param nets_this_round: 当前轮次中参与训练的客户端的本地模型字典 键 客户端索引 值 本地模型
    :param initial_global_parameters: 初始全局模型的参数 字典 键 baseconv1... 值 参数
    :param dw: 个客户端的本地模型参数与全局模型参数的差异 列表，每个元素字典，字典值baseconv1... 值初始为0
    :param similarity_matric: 相似度度量方式，例如余弦相似度 对角线元素相同-1 i,j == j,i
    :return: 模型差异矩阵
    """
    model_similarity_matrix = torch.zeros((len(nets_this_round), len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0),
                                                               weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    return model_similarity_matrix


def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, principal_list, dw,
                                 fed_avg_freqs, lambda_1,
                                 lambda_2, complementary_metric, similarity_metric):
    """
    更新客户端之间的协作图矩阵
    :param graph_matrix: 协作图矩阵，表示客户端之间的协作权重
    :param nets_this_round: 当前轮次中参与训练的客户端的本地模型字典 {0:model,1:model,...}
    :param initial_global_parameters: 初始全局模型的参数 {'base.conv1.weight':tensor,...}
    :param principal_list: 一个列表，包含每个客户端的主成分分析（PCA）结果，即正交基
    :param dw: 每个客户端的本地模型参数与全局模型参数的差异 [{'base.conv1.weight':tensor,...},{'base.conv1.weight':tensor,...}...]
    :param fed_avg_freqs: 每个客户端的数据量占比，用于联邦平均算法中的加权聚合 {0:proportion,1:proportion,...}
    :param lambda_1: 一个超参数，用于控制互补性矩阵的权重 0.9
    :param lambda_2: 一个超参数，用于控制差异性矩阵的权重 1.4
    :param complementary_metric: 一个字符串，表示用于计算互补性的度量方法 PA
    :param similarity_metric: 一个字符串，表示用于计算差异性的度量方法 all
    :return:  协作图矩阵
    """
    index_clientid = list(nets_this_round.keys())  # 获取参与训练的客户端索引
    model_complementary_matrix = cal_complementary(nets_this_round, principal_list, complementary_metric)  # 计算互补性矩阵
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw,
                                                          similarity_metric)  # 模型差异矩阵
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_complementary_matrix,
                                                    model_difference_matrix, lambda_1, lambda_2,
                                                    fed_avg_freqs)  # 优化协作图矩阵
    return graph_matrix


def cal_complementary(nets_this_round, principal_list, complementary_metric):
    """
    计算客户端模型之间的互补性矩阵
    :param nets_this_round: 当前轮次中参与训练的客户端的本地模型字典 {0:model,1:model,...}
    :param principal_list: 一个列表，包含每个客户端的主成分分析（PCA）结果，即正交基
    :param complementary_metric: 一个字符串，表示用于计算互补性的度量方法 PA
    :return: 互补性矩阵
    """
    model_complementary_matrix = np.zeros((len(nets_this_round), len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    k = principal_list[0].shape[0]  # 获取主成分的数量 k 3，即 principal_list 中每个数组的第一维大小 有客户端数量的(3,84)的二维数组
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if complementary_metric == "PA":
                phi = compute_principal_angles(principal_list[i], principal_list[j])  # 主成分角度
                # 互补性度量 主成分角度的平均余弦值 余弦值越接近 1，表示两个客户端的主成分越相似，互补性越低；余弦值越接近 0，表示互补性越高。
                principal_angle = np.cos((1 / k) * np.sum(phi))
                model_complementary_matrix[i][j] = principal_angle
                model_complementary_matrix[j][i] = principal_angle
    return model_complementary_matrix


def compute_principal_angles(A, B):
    """
    计算主成分角度
    :param A: 一个二维数组，表示第一个客户端的主成分矩阵，形状为 (k, d)，其中 k 是主成分的数量，d 是特征的维度。
    :param B: 一个二维数组，表示第二个客户端的主成分矩阵，形状为 (k, d)，其中 k 是主成分的数量，d 是特征的维度。
    :return: 主成分角度
    """
    assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"

    k = A.shape[0]  # 3
    # 计算归一化向量 A 和 B 的范数
    norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
    norm_B = np.linalg.norm(B, axis=1)
    # 计算 A 和 B 的内积
    dot_product = np.dot(A, B.T)
    # 计算余弦矩阵
    cosine_matrix = dot_product / (norm_A * norm_B)
    # 提取最大余弦值
    cos_phi_values = []

    for _ in range(k):
        i, j = np.unravel_index(np.argmax(cosine_matrix, axis=None), cosine_matrix.shape)  # 最大的索引位置
        cos_phi_values.append(cosine_matrix[i, j])  # 将最大的余弦值添加到列表中
        # 将最大的余弦值对应的行和列设置为 -inf，以便下一次循环时可以找到下一个最大的余弦值
        cosine_matrix[i, :] = -np.inf
        cosine_matrix[:, j] = -np.inf
    # 计算主成分角度 np.clip 将余弦值限制在 [-1, 1] 范围内
    phi = np.arccos(np.clip(cos_phi_values, -1, 1))

    return phi


def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_complementary_matrix, model_difference_matrix,
                                     lambda_1, lambda_2, fed_avg_freqs):
    """
    优化协作图矩阵
    :param graph_matrix:  当前的协作图矩阵，表示客户端之间的协作权重
    :param index_clientid: 当前轮次中参与训练的客户端的索引列表
    :param model_complementary_matrix: 客户端模型之间的互补性矩阵
    :param model_difference_matrix: 客户端模型之间的差异性矩阵
    :param lambda_1: 控制互补性矩阵的权重
    :param lambda_2: 控制差异性矩阵的权重
    :param fed_avg_freqs: 每个客户端的数据量占总数据量的比例，用于联邦平均（FedAvg）算法
    :return: 协作图矩阵
    """
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_complementary_vector = model_complementary_matrix[i]
        model_difference_vector = model_difference_matrix[i]
        s = model_difference_vector.numpy()
        c = model_complementary_vector
        q = lambda_1 * c + lambda_2 * s - 2 * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                          [G @ x <= h,
                           A @ x == b]
                          )
        prob.solve()

        graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
    return graph_matrix


def weight_flatten(model):
    params = []
    for k in model:
        if 'fc' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w):
    """
    根据协作图矩阵（graph_matrix）对客户端的模型参数进行聚合
    :param cfg:
    :param graph_matrix: 协作图矩阵
    :param nets_this_round: 当前轮次中参与训练的客户端的本地模型字典 键 客户端ID 值 对应模型
    :param global_w: 全局模型参数
    :return: 聚类模型字典
    """
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

        for neighbor_id in nets_this_round.keys():
            net_para = weight_flatten_all(nets_this_round[neighbor_id].state_dict())
            cluster_model_state += net_para * (aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para))

    for client_id in nets_this_round.keys():
        nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])

    return cluster_model_vectors


def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)


def compute_loss(net, test_data_loader):
    net.eval()
    loss, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            loss += torch.nn.functional.cross_entropy(out, target).item()
            total += x.data.size()[0]
    net.to('cpu')
    return loss / float(total)
