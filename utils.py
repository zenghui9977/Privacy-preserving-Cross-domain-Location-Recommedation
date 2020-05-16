import torch
import math
import numpy as np
import datetime
import os
import pickle
from metrics import *


def exe_time(func):
    def new_func(*args, **args2):
        name = func.__name__
        start = datetime.datetime.now()
        print("-- {%s} start: @ %ss" % (name, start))
        back = func(*args, **args2)
        end = datetime.datetime.now()
        print("-- {%s} end:   @ %ss" % (name, end))
        total = (end - start).total_seconds()
        print("-- {%s} total: @ %.3fs = %.3fh" % (name, total, total / 3600.0))
        return back
    return new_func




def combine_2_array(a, b):
    return np.append(a, b, axis=0)


def unique_the_list(list_with_some_same_ele):
    return np.array(list(set(tuple(t) for t in list_with_some_same_ele)))


# 计算坐标t[x,y]与t'[x',y']的欧氏距离
def euclidean_distance(t, t_):
    return np.sqrt(sum(np.power((t - t_), 2)))


# 计算Location t [x,y,c]与 t'[x',y',c']的语义欧氏距离
def semantic_euclidean_distance(t, t_):
    # if t.category == t'.category
    if t[2] == t_[2]:
        distance = euclidean_distance(t, t_)
    else:
        distance = np.inf
    return distance


# 依据交互数组构建交互矩阵
def build_interaction_matrix(user_num, item_num, interaction_pois):
    interaction_matrix = torch.zeros(user_num, item_num)
    for u in range(user_num):
        u_visited_poi = interaction_pois[u]
        for l in u_visited_poi:
            if l != item_num:
                interaction_matrix[u][l] += 1
    return interaction_matrix


def top_100_recommedated_item(predict_matrix, train_set_item_list):
    rec_pro, rec_index = predict_matrix.sort(1, descending=True)
    top_k = 100
    rec_top_k_list = []
    user_num, item_num = predict_matrix.size()
    for u in range(user_num):
        u_train_set = set(train_set_item_list[u])
        u_rec_index = rec_index[u].cpu().numpy()
        temp = []
        for i in range(item_num):
            if u_rec_index[i] not in u_train_set:
                temp.append(u_rec_index[i])
            if len(temp) == top_k:
                break
        rec_top_k_list.append(temp)

    return rec_top_k_list


def compute_metrics_each_epoch(user_num, item_num, rec_top_k_list, test_set):
    Top_K = [5, 10, 15, 20]

    precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num = [], [], [], [], []
    for top_k in Top_K:
        pre_list, rec_list, f_me_list, ndcg_list = [], [], [], []
        hit_num_list = 0
        for u in range(user_num):
            top_100_pred_list = rec_top_k_list[u]
            pred = top_100_pred_list[:top_k]
            gt = test_set[u]
            pre = precision(gt, pred)
            rec = recall(gt, pred)
            f_me = f_measure(gt, pred)
            ndcg = getNDCG(gt, pred)

            pre_list.append(pre)
            rec_list.append(rec)
            f_me_list.append(f_me)
            ndcg_list.append(ndcg)

            hit_num_list += hit_num_k(gt, pred)

        hit_num_list = hit_num_list / item_num
        print('top_k is %d' % top_k)
        print('[precision, recall, f_measure, NDCG, hit_num] --> [%f, %f, %f, %f, %f]' % (
        np.mean(pre_list), np.mean(rec_list), np.mean(f_me_list), np.mean(ndcg_list), hit_num_list))

        precision_mean.append(np.mean(pre_list))
        recall_mean.append(np.mean(rec_list))
        f_measure_mean.append(np.mean(f_me_list))
        ndcg_mean.append(np.mean(ndcg_list))
        hit_num.append(hit_num_list)

    return precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num

def save_as_csv(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savetxt(data_dir+file_name, data, delimiter=',')

def save_as_npy(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir+file_name, data)

def save_as_pkl(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filehandler = open(data_dir + "/" + filename + ".pkl", "wb")
    pickle.dump(data, filehandler)
    filehandler.close()

def save_as_pt(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    torch.save(data, data_dir + filename)

def read_from_npy_dict(data_dir, file_name):
    return np.load(data_dir + file_name).item()

def read_from_npy(data_dir, file_name):
    return np.load(data_dir + file_name)

def read_from_pkl(save_dir, filename):
    return pickle.load(open(save_dir + filename , 'rb'))

def read_from_pt(save_dir, filename):
    return torch.load(save_dir + filename)




@exe_time
def main():
    # poi_list = [[1, 1],[1, 2],[2, 1], [2, 2], [2,3], [3,2]]
    # item_num = len(poi_list)
    # alias_id_list = [0,1,2,3,4,5]
    # tree = kd_tree.create(poi_list)
    # user_iteraction = [[0,1],[1,3,4,5],[2,4],[0,3]]
    # # train_buys_mask, train_mask = full_data_buys_mask(user_iteraction, tail=[item_num])
    # # print(train_buys_mask, train_mask)
    # # print(fun_random_neg_masks_tra(item_num, train_buys_mask))
    # user_num = len(user_iteraction)
    # obf_user_iteraction = []
    # for u in range(user_num):
    #     user_visited_poi_cord = [poi_list[i] for i in user_iteraction[u]]
    #
    #     obs = geo_indistinguishability_obfuscation(user_visited_poi_cord, 1)
    #     remp = obfuscation_remapped(obs, tree)
    #     obf_user_iteraction.append(find_index_according_to_value(poi_list, remp))
    # obf_user_iteraction = [obf_user_iteraction[i].tolist() for i in range(len(obf_user_iteraction))]
    # print(obf_user_iteraction)
    # cul = compute_confidence_matrix(tree, obf_user_iteraction, 3, poi_list, 1)
    # print(cul)
    #
    # kkk=obfuscate_visited_data_list(user_iteraction, poi_list, tree, 1)
    # print(kkk)
    # interaction_matrix = build_interaction_matrix(user_num, item_num, obf_user_iteraction)
    # print(interaction_matrix)
    #
    # target_interaction = [[0,1],[2],[3],[1,2,3]]
    # target_user_num = len(target_interaction)
    # target_interaction_matrix = build_interaction_matrix(target_user_num, item_num, target_interaction)
    #
    # ccmf = CCMF(0.5, 0.5, 0.1, 0.01, 0.01, 10, user_num, target_user_num, item_num)
    epoch = 25
    # for e in range(epoch):
    #     ccmf.update(interaction_matrix, target_interaction_matrix, cul)
    #     print(ccmf.predict())


if '__main__' == __name__:
    main()