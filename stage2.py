# 本文件用于目标域的推荐以及指标评估

from utils import *
from load_data import *
from CCMF import *


PROCESS_DATA_FOLDER = 'process_data/Foursquare/'

# 获取目标域数据
item_aliases_list = read_from_pkl(PROCESS_DATA_FOLDER, 'item_aliases_list.pkl')
user_item_num = read_from_pkl(PROCESS_DATA_FOLDER, 'user-item_num.pkl')

user_num, item_num = user_item_num[0], user_item_num[1]
_, target_domain_train_data, target_domain_test_data = split_orgina_data(item_aliases_list)

# 目标域数据负采样
target_domain_train_data_with_mask, target_domain_train_data_mask = full_data_buys_mask(target_domain_train_data, tail=[item_num])
target_domain_test_data_with_mask, target_domain_test_data_mask = full_data_buys_mask(target_domain_test_data, tail=[item_num])

target_domain_train_data_neg_sample_list = fun_random_neg_masks_tra(item_num, target_domain_train_data_with_mask)
target_domain_test_data_neg_sample_list = fun_random_neg_masks_tes(item_num, target_domain_train_data_with_mask, target_domain_test_data_with_mask)

combined_target_domain_train_data = combine_2_array(target_domain_train_data_with_mask, target_domain_train_data_neg_sample_list)
combined_target_domain_test_data = combine_2_array(target_domain_test_data_with_mask, target_domain_test_data_neg_sample_list)

target_domain_user_num = len(combined_target_domain_train_data)

# 读取加噪的辅助域数据以及置信矩阵
obfuscated_auxiliary_domain_data = read_from_pkl(PROCESS_DATA_FOLDER, 'obfuscated_auxiliary_domain_data.pkl')
confidence_matrix = read_from_npy(PROCESS_DATA_FOLDER, 'confidence_matrix.npy')

auxiliary_domain_user_num = len(obfuscated_auxiliary_domain_data)

# 依据数据构建输入矩阵（真实值），datatype = Tensor
auxiliary_data = build_interaction_matrix(auxiliary_domain_user_num, item_num, obfuscated_auxiliary_domain_data)
target_data = build_interaction_matrix(target_domain_user_num, item_num, combined_target_domain_train_data)


# 参数设置
epoch = 25
w_a = 0.5
w_t = 0.5
learning_rate = 0.01
lamda_Q = 0.1
lamda_P = 0.1
k = 20
ccmf = CCMF(w_a, w_t, learning_rate, lamda_Q, lamda_P, k, auxiliary_domain_user_num, target_domain_user_num, item_num)


precision_each_round, recall_each_round, f_measure_each_round, ndcg_each_round, hit_num_each_round = [], [], [], [], []

for e in range(epoch):
    ccmf.update(auxiliary_data, target_data, confidence_matrix)
    pred = ccmf.predict()
    print(pred[0])
    rec_top_k_list = top_100_recommedated_item(pred, combined_target_domain_train_data)
    print(rec_top_k_list[0][:20])
    precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num = compute_metrics_each_epoch(target_domain_user_num, item_num, rec_top_k_list, combined_target_domain_test_data)

    precision_each_round.append(precision_mean)
    recall_each_round.append(recall_mean)
    f_measure_each_round.append(f_measure_mean)
    ndcg_each_round.append(ndcg_mean)
    hit_num_each_round.append(hit_num)
    print('********************')



print('the statistical information in each federated learning:')
print('\t the best performance of top@[5, 10, 15, 20]')
print('\t \t \t \t round \t \t metrics value')
print('\t \t precision \t %s ---> %s' % (np.argmax(precision_each_round, axis=0), max(precision_each_round)))
print('\t \t recall \t %s ---> %s' % (np.argmax(recall_each_round, axis=0), max(recall_each_round)))
print('\t \t f_measure \t %s ---> %s' % (np.argmax(f_measure_each_round, axis=0), max(f_measure_each_round)))
print('\t \t NDCG \t \t %s ---> %s' % (np.argmax(ndcg_each_round, axis=0), max(ndcg_each_round)))
print('\t \t hit_num \t %s ---> %s' % (np.argmax(hit_num_each_round, axis=0), max(hit_num_each_round)))