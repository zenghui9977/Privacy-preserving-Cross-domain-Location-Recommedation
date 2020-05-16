from load_data import *
from utils import *
import kd_tree
from Geo_indistinguishability import *
from CCMF import *

DATA_FOLDER = 'data/'
FILE_NAME = 'Foursquare.txt'

PROCESS_DATA_FOLDER = 'process_data/Foursquare/'

epsilon = 1
m = 3

# # 读取数据
# user_num, item_num, item_aliases_dict, user_aliases_dict, item_aliases_list, user_aliases_list, location = load_data(DATA_FOLDER + FILE_NAME)
# location = [list(i) for i in location]
#
#
# save_as_npy(PROCESS_DATA_FOLDER, 'item_aliases_dict.npy', item_aliases_dict)
# save_as_npy(PROCESS_DATA_FOLDER, 'user_aliases_dict.npy', user_aliases_dict)
# save_as_pkl(PROCESS_DATA_FOLDER, 'item_aliases_list', item_aliases_list)
# save_as_pkl(PROCESS_DATA_FOLDER, 'user_aliases_list', user_aliases_list)
# save_as_pkl(PROCESS_DATA_FOLDER, 'location', location)
# save_as_pkl(PROCESS_DATA_FOLDER, 'user-item_num', [user_num, item_num])

# # 切分辅助域 、目标域、测试集
# print('切分辅助域 、目标域、测试集')
# auxiliary_domain_data, target_domain_train_data, target_domain_test_data = split_orgina_data(item_aliases_list)
#
# # 构建KD-tree
# print('构建KD-tree')
# tree = kd_tree.create(location)
#
# # 辅助域数据加噪并计算置信矩阵
# print('辅助域数据加噪')
# obfuscated_auxiliary_domain_data = obfuscate_visited_data_list(auxiliary_domain_data, location, tree, epsilon=epsilon)
# print('计算置信矩阵')
# confidence_matrix = compute_confidence_matrix(tree, obfuscated_auxiliary_domain_data, m, location, epsilon)
#
# save_as_pkl(PROCESS_DATA_FOLDER, 'obfuscated_auxiliary_domain_data', obfuscated_auxiliary_domain_data)
# save_as_npy(PROCESS_DATA_FOLDER, 'confidence_matrix.npy', confidence_matrix)


user_item_num = read_from_pkl(PROCESS_DATA_FOLDER, 'user-item_num.pkl')
item_aliases_list = read_from_pkl(PROCESS_DATA_FOLDER, 'item_aliases_list.pkl')
obfuscated_auxiliary_domain_data = read_from_pkl(PROCESS_DATA_FOLDER, 'obfuscated_auxiliary_domain_data.pkl')
confidence_matrix = read_from_npy(PROCESS_DATA_FOLDER, 'confidence_matrix.npy')
location = read_from_pkl(PROCESS_DATA_FOLDER, 'location.pkl')


user_num, item_num = user_item_num[0], user_item_num[1]
auxiliary_domain_data, target_domain_train_data, target_domain_test_data = split_orgina_data(item_aliases_list)

# 负采样增加训练数据
print('负采样增加训练数据')
auxiliary_domain_data_with_mask, auxiliary_domain_data_mask = full_data_buys_mask(auxiliary_domain_data, tail=[item_num])
target_domain_train_data_with_mask, target_domain_train_data_mask = full_data_buys_mask(target_domain_train_data, tail=[item_num])
target_domain_test_data_with_mask, target_domain_test_data_mask = full_data_buys_mask(target_domain_test_data, tail=[item_num])

auxiliary_domain_data_neg_sample_list = fun_random_neg_masks_tra(item_num, auxiliary_domain_data_with_mask)
target_domain_train_data_neg_sample_list = fun_random_neg_masks_tra(item_num, target_domain_train_data_with_mask)
target_domain_test_data_neg_sample_list = fun_random_neg_masks_tes(item_num, target_domain_train_data_with_mask, target_domain_test_data_with_mask)

combined_auxiliary_domain_data = combine_2_array(auxiliary_domain_data_with_mask, auxiliary_domain_data_neg_sample_list)
combined_target_domain_train_data = combine_2_array(target_domain_train_data_with_mask, target_domain_train_data_neg_sample_list)
combined_target_domain_test_data = combine_2_array(target_domain_test_data_with_mask, target_domain_test_data_neg_sample_list)


#
# # 构建KD-tree
# print('构建KD-tree')
# tree = kd_tree.create(location)
#
# # 对负采样之后的辅助域数据进行加噪
# print('辅助域数据加噪')
# obfuscated_auxiliary_domain_data = obfuscate_visited_data_list(combined_auxiliary_domain_data, location, tree, epsilon=epsilon)
# print('计算置信矩阵')
# confidence_matrix = compute_confidence_matrix(tree, obfuscated_auxiliary_domain_data, m, location, epsilon)
#
#
# save_as_pkl(PROCESS_DATA_FOLDER, 'obfuscated_auxiliary_domain_data', obfuscated_auxiliary_domain_data)
# save_as_npy(PROCESS_DATA_FOLDER, 'confidence_matrix.npy', confidence_matrix)


obfuscated_auxiliary_domain_data = read_from_pkl(PROCESS_DATA_FOLDER, 'obfuscated_auxiliary_domain_data.pkl')
auxiliary_domain_user_num = len(obfuscated_auxiliary_domain_data)
target_domain_user_num = len(combined_target_domain_train_data)
print(auxiliary_domain_user_num, target_domain_user_num)

auxiliary_data = build_interaction_matrix(auxiliary_domain_user_num, item_num, obfuscated_auxiliary_domain_data)
target_data = build_interaction_matrix(target_domain_user_num, item_num, combined_target_domain_train_data)


w_a = 0.5
w_t = 0.5
learning_rate = 0.01
lamda_Q = 0.1
lamda_P = 0.1
k = 20
ccmf = CCMF(w_a, w_t, learning_rate, lamda_Q, lamda_P, k, auxiliary_domain_user_num, target_domain_user_num, item_num)

precision_each_round, recall_each_round, f_measure_each_round, ndcg_each_round, hit_num_each_round = [], [], [], [], []
epoch = 25
for e in range(epoch):
    ccmf.update(auxiliary_data, target_data, confidence_matrix)
    pred = ccmf.predict()
    print(pred[0])
    rec_top_k_list = top_100_recommedated_item(pred, combined_target_domain_train_data)
    print(rec_top_k_list[0][:20])
    precision_mean, recall_mean, f_measure_mean, ndcg_mean, hit_num = compute_metrics_each_epoch(target_domain_user_num, item_num, rec_top_k_list, combined_target_domain_test_data)
    print('********************')
    precision_each_round.append(precision_mean)
    recall_each_round.append(recall_mean)
    f_measure_each_round.append(f_measure_mean)
    ndcg_each_round.append(ndcg_mean)
    hit_num_each_round.append(hit_num)




print('the statistical information in each federated learning:')
print('\t the best performance of top@[5, 10, 15, 20]')
print('\t \t \t \t round \t \t metrics value')
print('\t \t precision \t %s ---> %s' % (np.argmax(precision_each_round, axis=0), max(precision_each_round)))
print('\t \t recall \t %s ---> %s' % (np.argmax(recall_each_round, axis=0), max(recall_each_round)))
print('\t \t f_measure \t %s ---> %s' % (np.argmax(f_measure_each_round, axis=0), max(f_measure_each_round)))
print('\t \t NDCG \t \t %s ---> %s' % (np.argmax(ndcg_each_round, axis=0), max(ndcg_each_round)))
print('\t \t hit_num \t %s ---> %s' % (np.argmax(hit_num_each_round, axis=0), max(hit_num_each_round)))