# 本程序用于对辅助域数据进行加噪，计算相应的置信矩阵，并存储数据，此过程耗时约2.5h
from utils import *
from load_data import *
from Geo_indistinguishability import *
import os

DATA_FOLDER = 'data/'
FILE_NAME = 'Foursquare.txt'

PROCESS_DATA_FOLDER = 'process_data/Foursquare/'

# 读取数据,读取一次将文件存储起来
if not os.path.exists(PROCESS_DATA_FOLDER + 'item_aliases_dict.npy'):
    print('读取原始数据')
    user_num, item_num, item_aliases_dict, user_aliases_dict, item_aliases_list, user_aliases_list, location = load_data(DATA_FOLDER + FILE_NAME)
    location = [list(i) for i in location]

    save_as_npy(PROCESS_DATA_FOLDER, 'item_aliases_dict.npy', item_aliases_dict)
    save_as_npy(PROCESS_DATA_FOLDER, 'user_aliases_dict.npy', user_aliases_dict)
    save_as_pkl(PROCESS_DATA_FOLDER, 'item_aliases_list', item_aliases_list)
    save_as_pkl(PROCESS_DATA_FOLDER, 'user_aliases_list', user_aliases_list)
    save_as_pkl(PROCESS_DATA_FOLDER, 'location', location)
    save_as_pkl(PROCESS_DATA_FOLDER, 'user-item_num', [user_num, item_num])

# 读取保存的数据
print('读取处理后的数据')
user_item_num = read_from_pkl(PROCESS_DATA_FOLDER, 'user-item_num.pkl')
item_aliases_list = read_from_pkl(PROCESS_DATA_FOLDER, 'item_aliases_list.pkl')
location = read_from_pkl(PROCESS_DATA_FOLDER, 'location.pkl')

user_num, item_num = user_item_num[0], user_item_num[1]
auxiliary_domain_data, target_domain_train_data, target_domain_test_data = split_orgina_data(item_aliases_list)


# 构建KD-tree
print('构建KD-tree')
tree = kd_tree.create(location)

# 对辅助域数据进行负采样
print('对辅助域数据进行负采样')
auxiliary_domain_data_with_mask, auxiliary_domain_data_mask = full_data_buys_mask(auxiliary_domain_data, tail=[item_num])
auxiliary_domain_data_neg_sample_list = fun_random_neg_masks_tra(item_num, auxiliary_domain_data_with_mask)
combined_auxiliary_domain_data = combine_2_array(auxiliary_domain_data_with_mask, auxiliary_domain_data_neg_sample_list)

# 设置参数
m = 3           # 置信矩阵的计算中，选取最近的m个坐标
epsilon = 1     # privacy level隐私等级


# 辅助域数据加噪并计算置信矩阵
print('辅助域数据加噪')
obfuscated_auxiliary_domain_data = obfuscate_visited_data_list(auxiliary_domain_data, location, tree, epsilon=epsilon)
print('计算置信矩阵')
confidence_matrix = compute_confidence_matrix(tree, obfuscated_auxiliary_domain_data, m, location, epsilon)

save_as_pkl(PROCESS_DATA_FOLDER, 'obfuscated_auxiliary_domain_data', obfuscated_auxiliary_domain_data)
save_as_npy(PROCESS_DATA_FOLDER, 'confidence_matrix.npy', confidence_matrix)