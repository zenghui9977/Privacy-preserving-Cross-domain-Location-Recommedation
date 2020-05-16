import pandas as pd
import numpy as np
import random


def load_data(dataset_dir):
    print('Loading original data ......')
    data = pd.read_csv(dataset_dir, sep=' ')

    all_user_pois = [[i for i in u_pois.split('/')] for u_pois in data['u_pois']]
    all_user_cordi = [[i.split(',') for i in u_pois.split('/')] for u_pois in data['u_coordinates']]


    all_users = [i for i in data['u_id']]

    all_pois = [i for u_pois in all_user_pois for i in u_pois]
    all_cordis = [i for u_cordis in all_user_cordi for i in u_cordis]




    all_items = set(all_pois)

    user_num, item_num = len(all_users), len(all_items)
    print('\tusers, items:  = {v1}, {v2}'.format(v1=user_num, v2=item_num))

    # 构建重新映射图
    item_aliases_dict = dict(zip(all_items, range(item_num)))
    user_aliases_dict = dict(zip(all_users, range(user_num)))

    # 依据重映射图，进行重映射
    item_aliases_list = [[item_aliases_dict[i] for i in item] for item in all_user_pois]
    user_aliases_list = [user_aliases_dict[i] for i in all_users]

    poi_map_cordis = dict(zip(all_pois, all_cordis))
    cordi_new = dict()
    for i in poi_map_cordis:
        cordi_new[item_aliases_dict[i]] = poi_map_cordis[i]
    location = []
    for i in range(len(poi_map_cordis)):
        location.append(cordi_new[i])
    location = np.asarray(location, dtype='float')


    del poi_map_cordis,cordi_new

    return user_num, item_num, \
           item_aliases_dict, user_aliases_dict, \
           item_aliases_list, user_aliases_list, location


# auxiliary domain, target_domain, test_data ----》 70%， 30%， 1
def split_orgina_data(data):
    data_num = len(data)
    auxiliary_domain_data_num = int(data_num * 0.7)
    target_domain_data_num = data_num - auxiliary_domain_data_num

    auxiliary_domain_data = data[:auxiliary_domain_data_num]
    target_domain_data = data[auxiliary_domain_data_num:]


    # target domain split train data and test data
    target_domain_train_data, target_domain_test_data = [], []
    for u in range(target_domain_data_num):
        the_user_data = target_domain_data[u]

        the_user_train_data = the_user_data[:-1]
        the_user_test_data = [the_user_data[-1]]

        target_domain_train_data.append(the_user_train_data)
        target_domain_test_data.append(the_user_test_data)

    return auxiliary_domain_data, target_domain_train_data, target_domain_test_data


def full_data_buys_mask(all_user_buys, tail):
    # 将train/test中序列补全为最大长度，补的idx值=item_num. 为了能在theano里对其进行shared。
    # tail, 添加的。商品索引是0~item_num-1，所以该值[item_num]是没有对应商品实物的。
    us_lens = [len(ubuys) for ubuys in all_user_buys]
    len_max = max(us_lens)
    us_buys = [ubuys + tail * (len_max - le) for ubuys, le in zip(all_user_buys, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_buys, us_msks


def fun_random_neg_masks_tra(item_num, tras_mask):
    """
    从num件商品里随机抽取与每个用户的购买序列等长且不在已购买商品里的标号。后边补全的负样本用虚拟商品[item_num]
    """
    us_negs = []
    for utra in tras_mask:     # 每条用户序列
        unegs = []
        for i, e in enumerate(utra):
            if item_num == e:                    # 表示该购买以及之后的，都是用虚拟商品[item_num]来补全的
                unegs += [item_num] * (len(utra) - i)   # 购买序列里对应补全商品的负样本也用补全商品表示
                break
            j = random.randint(0, item_num - 1)  # 负样本在商品矩阵里的标号
            while j in utra:                     # 抽到的不是用户训练集里的。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs


def fun_random_neg_masks_tes(item_num, tras_mask, tess_mask):
    """
    从num件商品里随机抽取与测试序列等长且不在训练序列、也不再测试序列里的标号
    """
    us_negs = []
    for utra, utes in zip(tras_mask, tess_mask):
        unegs = []
        for i, e in enumerate(utes):
            if item_num == e:                   # 尾部补全mask
                unegs += [item_num] * (len(utes) - i)
                break
            j = random.randint(0, item_num - 1)
            while j in utra or j in utes:         # 不在训练序列，也不在预测序列里。
                j = random.randint(0, item_num - 1)
            unegs += [j]
        us_negs.append(unegs)
    return us_negs