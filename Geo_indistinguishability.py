import torch
import numpy as np
import math
from scipy.special import lambertw
import kd_tree
import datetime


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


# poi_list: 所有原始辅助域坐标，每个坐标结构为[x, y]
# epsilon: 隐私保护水平
def geo_indistinguishability_obfuscation(poi_list, epsilon):
    # 获取辅助域包含的数据量
    poi_num = len(poi_list)

    # 服从均匀分布的变量 theta, p
    # theta --> [0,1]  p --> [0, 2 * PI]
    theta = np.random.uniform(0, 1, poi_num)
    p = np.random.uniform(0, 2 * math.pi, poi_num)

    r = [- 1 / epsilon * (lambertw((p_ - 1) / math.e, -1) + 1) for p_ in p ]
    # r返回形式为complex，其虚部为0，将其转换成实数
    noise = np.dstack((np.real(r) * np.cos(theta), np.real(r) * np.sin(theta))).reshape((poi_num, -1))
    # 将原始坐标(前两列)与噪声相加，加噪阶段
    obfuscated_poi = np.array(poi_list) + noise

    return obfuscated_poi


def obfuscation_remapped(obfuscated_poi, kdtree):
    remap_poi = []
    for poi in obfuscated_poi:
        kdnode, distance = kdtree.search_nn(poi)
        remap_node_poi = kdnode.data
        remap_poi.append(remap_node_poi)
    return remap_poi

@exe_time
def obfuscate_visited_data_list(visited_poi_list, location, kdtree, epsilon):
    user_num = len(visited_poi_list)
    item_num = len(location)
    obfuscate_poi_list=[]
    for u in range(user_num):
        user_visited_poi_cord = [location[i] for i in visited_poi_list[u] if i<item_num]
        obfuscated_poi_cord = geo_indistinguishability_obfuscation(user_visited_poi_cord, epsilon)
        remapped_poi_cord = obfuscation_remapped(obfuscated_poi_cord, kdtree)
        obfuscate_poi_list.append(list(find_index_according_to_value(location, remapped_poi_cord)))

    return obfuscate_poi_list

'''
寻找search数组中对应basic数组的下标
basic_array: 原数组
search_array: 需要搜寻的数据
'''
def find_index_according_to_value(basic_array, search_array):
    basic_array = np.asarray(basic_array)
    search_array = np.asarray(search_array)
    return np.where((basic_array == search_array[:,None]).all(-1))[1]


def compute_cul(distance_list, epsilon):
    e_distance_list = [np.exp(-epsilon * d) for d in distance_list]
    cul = e_distance_list / sum(e_distance_list)

    return cul


'''
kdtree: 用于寻找离t'最近的m个节点
obfuscated_visited_poi: 所有用户加噪后每个节点的交互数据的集合[user1:[x1',x2'...xn'],user2:[y1',...ym'],...usern:[z1',...,zk']]
m: 选取加噪位置最近的m个poi
location： 坐标数组
'''
def compute_confidence_matrix(kdtree, obfuscated_visited_poi, m, location, epsilon):
    print('wo bu hui a ! tai nan le')
    user_num, poi_num = len(obfuscated_visited_poi), len(location)
    confidence_matrix = np.zeros((user_num, poi_num))
    for u in range(user_num):
        user_visited_pois = obfuscated_visited_poi[u]
        for u_visited_poi in user_visited_pois:
            nearest_m_node = kdtree.search_knn(location[u_visited_poi], m)
            nearest_m_cordi = [nearest_m_node[i][0].data for i in range(m)]
            nearest_m_distance = [nearest_m_node[i][1] for i in range(m)]

            nearest_m_poi_id = find_index_according_to_value(location, nearest_m_cordi)
            cul = compute_cul(nearest_m_distance, epsilon)
            for i in range(m):
                if confidence_matrix[u, nearest_m_poi_id[i]] < cul[i]:
                    confidence_matrix[u, nearest_m_poi_id[i]] = cul[i]

    return confidence_matrix

