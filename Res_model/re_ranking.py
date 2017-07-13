#!/usr/bin/python2
# -*- coding: utf-8 -*-

'''
Re-ranking Person Re-identification with k-reciprocal Encoding
'''

import numpy as np
from scipy.spatial.distance import cdist


def get_name_feature_dict(features, names):
    '''
    :param features: gallery_num * feature_dim
    :param names: gallery_num
    :return: name_feature_dict
    '''
    name_feature_dict = {}
    for i in range(features.shape[0]):
        name_feature_dict[names[i]] = features[i]
    return name_feature_dict


def get_name_order_dict(names):
    '''
    :param names: name num
    :return: name_order_dict
    '''
    name_order_dict = {}
    i = 0
    for name in names:
        name_order_dict[name] = i
        i += 1
    return name_order_dict


def N(feature, p_feature, p_name, g_features, g_names, k):
    '''
    :param feature: feature_dim
    :param p_feature: feature_dim
    :param p_name: int
    :param g_features: gallery_num * feature_dim
    :param g_names: gallery_num
    :param k: k1 nearest
    :return: k1 nearest neighbors' name list
    '''
    pg_feature = np.concatenate((np.array([p_feature]), g_features), axis=0)
    pg_name = np.concatenate((np.array([p_name]), g_names), axis=0)
    distmat = cdist(np.array([feature]), pg_feature)  # 1 * (1 + gallery_num)
    order = np.argsort(distmat, axis=1)
    return pg_name[order][0][1:k + 1].tolist()


def R(feature, feature_name, p_feature, g_features, g_names, pg_name_feature_dict, k):
    '''
    :param feature: feature_dim
    :param feature_name: int
    :param p_feature: feature_dim
    :param g_features: gallery_num * feature_dim
    :param g_names: gallery_num
    :param pg_name_feature_dict: key-gallery name, value-gallery feature
    :param k: k1 nearest
    :return: k1-reciprocal nearest neighbors' name set
    '''
    g_near_names = N(feature, p_feature, -1, g_features, g_names, k)
    for g_near_name in g_near_names:
        if feature_name not in N(pg_name_feature_dict[g_near_name], p_feature, -1, g_features, g_names, k):
            g_near_names.remove(g_near_name)
    return set(g_near_names)


def R_star(feature, feature_name, p_feature, g_features, g_names, pg_name_feature_dict, k):
    '''
    :param feature: feature_dim
    :param feature_name: int
    :param p_feature: feature_dim
    :param g_features: gallery_num * feature_dim
    :param g_names: gallery_num
    :param pg_name_feature_dict: key-gallery name, value-gallery feature
    :param k: k1 nearest
    :return: re-calculate k1-reciprocal nearest neighbors' name set
    '''
    R_pk_set = R(feature, feature_name, p_feature, g_features, g_names, pg_name_feature_dict, k)
    # print R_pk_set
    for q in R_pk_set:
        R_q12k_set = R(pg_name_feature_dict[q], q, p_feature, g_features, g_names, pg_name_feature_dict, int(k / 2))
        if float(len(R_pk_set & R_q12k_set)) >= 2.0 / 3.0 * len(R_q12k_set):
            R_pk_set = R_pk_set | R_q12k_set
    if -1 in R_pk_set:
        R_pk_set.remove(-1)
    return R_pk_set


def get_mahe_distance(feature1, feature2):
    '''
    :param feature1: feature_dim
    :param feature2: feature_dim
    :return: Mahalanobis distance
    '''
    # np.exp(-np.sum((feature1 - feature2) ** 2)) # this should be Mahalanobis distance, but it's too big
    return np.exp(-np.mean((feature1 - feature2) ** 2))  # average Mahalanobis distance on features


def get_jaccard_distance(p_feature, g_names, g_features, k1):
    '''
    :param p_feature: feature_dim
    :param g_names: gallery_num
    :param g_features: gallery_num * feature_dim
    :param k1: k1 nearest
    :return: Jaccard distance
    '''
    g_name_order_dict = get_name_order_dict(g_names)
    pg_name_feature_dict = get_name_feature_dict(g_features, g_names)  # no p name and feature
    pg_name_feature_dict[-1] = p_feature  # add p name and feature

    # get R_star_pk_name_set
    R_star_pk_name_set = R_star(p_feature, -1, p_feature, g_features, g_names, pg_name_feature_dict, k=k1)

    # get V_gis
    V_gi_list = []
    for gi_name in R_star_pk_name_set:
        V_gi = np.zeros(g_features.shape[0])
        R_star_gik_name_set = R_star(pg_name_feature_dict[gi_name], gi_name, p_feature, g_features, g_names,
                                     pg_name_feature_dict, k=k1)
        for gj_name in R_star_gik_name_set:
            V_gi[g_name_order_dict[gj_name]] = get_mahe_distance(pg_name_feature_dict[gj_name],
                                                                 pg_name_feature_dict[gi_name])
        V_gi_list.append(V_gi)

    # get V_p without local query expansion, implement as function (7) in literature
    V_p = np.zeros(g_features.shape[0])
    for gj_name in R_star_pk_name_set:
        V_p[g_name_order_dict[gj_name]] = get_mahe_distance(pg_name_feature_dict[gj_name], p_feature)

    # Note:
    # I don't know how to compute V_p with local query expansion as literature function (11).
    # Actually, I think local query expansion has been implement in R_star().

    # compute Jaccard distance
    V_min_sum = np.zeros(g_features.shape[0])
    V_max_sum = np.zeros(g_features.shape[0])
    for V_gi in V_gi_list:
        V_min_sum += np.minimum(V_p, V_gi)
        V_max_sum += np.maximum(V_p, V_gi)
    djpg = 1 - V_min_sum / (V_max_sum + 1e-12)
    return djpg


def re_ranking(distmat, p_features, g_names, g_features, k1, lambda_, top_n):
    '''
    :param distmat: probe_num * gallery_num
    :param p_features: probe_num * feature_dim
    :param g_names: gallery_num
    :param g_features: gallery_num * feature_dim
    :param k1: float
    :param lambda_: float
    :param top_n: float
    :return: distance with re-ranking
    '''
    for i in range(p_features.shape[0]):
        dist_pg = distmat[i]
        orders = np.argsort(dist_pg)
        g_names_sort = g_names[orders]

        jdist_pg_sort_top_n = get_jaccard_distance(p_features[i], g_names_sort[:top_n], g_features[orders[:top_n]],
                                                   k1=k1)
        jdist_pg = np.ones(dist_pg.shape)
        for j, order in enumerate(orders[:top_n]):
            jdist_pg[order] = jdist_pg_sort_top_n[j]
        dist_star_pg = (1.0 - lambda_) * jdist_pg + lambda_ * dist_pg
        distmat[i] = dist_star_pg
    return distmat


def test():
    print('generate_predict_xml(normalize_flag=%s, contain_top_n=%s)')
    g_features = np.load('result/test_features/predict_gallery_features.npy')
    p_features = np.load('result/test_features/predict_probe_features.npy')
    g_names_list = []
    with open('../data/predict_gallery_name.csv', "r") as f:
        for name_order in f.readlines():
            name_order = name_order.strip('\n')
            g_names_list.append(int(name_order.split(',')[0]))
    g_names = np.array(g_names_list)
    p_names_list = []
    with open('../data/predict_probe_name.csv', "r") as f:
        for name_order in f.readlines():
            name_order = name_order.strip('\n')
            p_names_list.append(int(name_order.split(',')[0]))
    p_names = np.array(p_names_list)
    distmat = cdist(g_features, p_features)
    re_ranking(distmat.T, p_features, g_names, g_features, 20, 0.3, 100)


if __name__ == '__main__':
    test()
