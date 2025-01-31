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
    model_similarity_matrix = torch.zeros((len(nets_this_round),len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] =  model_i[key] - initial_global_parameters[key]
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif  similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0), weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    return model_similarity_matrix




def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, principal_list, dw, fed_avg_freqs, lambda_1, 
                                    lambda_2, complementary_metric, similarity_metric):
    index_clientid = list(nets_this_round.keys())
    model_complementary_matrix = cal_complementary(nets_this_round, principal_list, complementary_metric)
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_metric)
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_complementary_matrix, model_difference_matrix, lambda_1, lambda_2, fed_avg_freqs)
    return graph_matrix

def cal_complementary(nets_this_round, principal_list, complementary_metric):
    model_complementary_matrix = np.zeros((len(nets_this_round),len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    k = principal_list[0].shape[0]
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if complementary_metric == "PA":
                phi = compute_principal_angles(principal_list[i], principal_list[j])
                principal_angle = np.cos((1 / k) * np.sum(phi))
                model_complementary_matrix[i][j] = principal_angle
                model_complementary_matrix[j][i] = principal_angle
    return model_complementary_matrix

def compute_principal_angles(A, B):
    assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"
    
    k = A.shape[0]
    norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
    norm_B = np.linalg.norm(B, axis=1)
    dot_product = np.dot(A, B.T)
    cosine_matrix = dot_product / (norm_A * norm_B)
    cos_phi_values = []

    for _ in range(k):
        i, j = np.unravel_index(np.argmax(cosine_matrix, axis=None), cosine_matrix.shape)
        cos_phi_values.append(cosine_matrix[i, j])
        cosine_matrix[i, :] = -np.inf
        cosine_matrix[:, j] = -np.inf
    phi = np.arccos(np.clip(cos_phi_values, -1, 1))

    return phi

def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_complementary_matrix, model_difference_matrix, lambda_1, lambda_2, fed_avg_freqs):
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
        q = lambda_1*c + lambda_2*s - 2 * p
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

