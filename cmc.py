#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.spatial.distance import cdist
import numpy.matlib as matlib
import cPickle as pickle
from xml.dom.minidom import Document
import os
import matplotlib.pyplot as plt

def _cmc_core(D, G, P):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)  # G[order],按距离排序的矩阵，第一行为top1,从左到右为各plabel对应的glabel
    return (match.sum(axis=1) * 1.0 / n).cumsum()  # 从上至下累积求和


def count(distmat, glabels=None, plabels=None, n_selected_labels=None, n_repeat=100):
    """Compute the Cumulative Match Characteristic(CMC)
    Args:
        distmat: A ``m×n`` distance matrix. ``m`` and ``n`` are the number of
            gallery and probe samples, respectively. In the case of ``glabels``
            and ``plabels`` both are ``None``, we assume both gallery and probe
            samples to be unique, i.e., the i-th gallery samples matches only to
            the i-th probe sample.
        glabels: Vector of length ``m`` that represents the labels of gallery
            samples
        plabels: Vector of length ``n`` that represents the labels of probe
            samples
        n_selected_labels: If specified, we will select only part of all the
            labels to compute the CMC.
        n_repeat: The number of random sampling times
    Returns:
        A vector represents the average CMC
    """

    m, n = distmat.shape

    if glabels is None and plabels is None:
        glabels = np.arange(0, m)
        plabels = np.arange(0, n)

    if type(glabels) is list:
        glabels = np.asarray(glabels)
    if type(plabels) is list:
        plabels = np.asarray(plabels)

    unique_glabels = np.unique(glabels)

    if n_selected_labels is None:
        n_selected_labels = unique_glabels.size

    ret = 0
    for r in range(n_repeat):
        # Randomly select gallery labels
        ind = np.random.choice(unique_glabels.size,
                               n_selected_labels,
                               replace=False)
        ind.sort()
        g = unique_glabels[ind]

        # Select corresponding probe samples
        ind = []
        for i, label in enumerate(plabels):
            if label in g: ind.append(i)
        ind = np.asarray(ind)

        p = plabels[ind]

        # Randomly select one sample per selected label
        subdist = np.zeros((n_selected_labels, p.size))
        for i, glabel in enumerate(g):
            samples = np.where(glabels == glabel)[0]
            j = np.random.choice(samples)
            subdist[i, :] = distmat[j, ind]

        # Compute CMC
        ret += _cmc_core(subdist, g, p)

    return ret / n_repeat


def count_lazy(distfunc, glabels=None, plabels=None, n_selected_labels=None, n_repeat=100):
    """Compute the Cumulative Match Characteristic(CMC) in a lazy manner
    This function will only compute the distance when needed.
    Args:
        distfunc: A distance computing function. Denote the number of gallery
            and probe samples by ``m`` and ``n``, respectively.
            ``distfunc(i, j)`` should output distance between gallery sample
            ``i`` and probe sample ``j``. In the case of ``glabels``
            and ``plabels`` both are integers, ``m`` should be equal to ``n``
            and we assume both gallery and probe samples to be unique,
            i.e., the i-th gallery samples matches only to the i-th probe
            sample.
        glabels: Vector of length ``m`` that represents the labels of gallery
            samples. Or an integer ``m``.
        plabels: Vector of length ``n`` that represents the labels of probe
            samples. Or an integer ``n``.
        n_selected_labels: If specified, we will select only part of all the
            labels to compute the CMC.
        n_repeat: The number of random sampling times
    Returns:
        A vector represents the average CMC
    """

    if type(glabels) is int:
        m = glabels
        glabels = np.arange(0, m)
    elif type(glabels) is list:
        glabels = np.asarray(glabels)
        m = glabels.size
    else:
        m = glabels.size

    if type(plabels) is int:
        n = plabels
        plabels = np.arange(0, n)
    elif type(plabels) is list:
        plabels = np.asarray(plabels)
        n = plabels.size
    else:
        n = plabels.size

    unique_glabels = np.unique(glabels)

    if n_selected_labels is None:
        n_selected_labels = unique_glabels.size

    ret = 0
    for r in range(n_repeat):
        # Randomly select gallery labels
        ind = np.random.choice(unique_glabels.size,
                               n_selected_labels,
                               replace=False)
        ind.sort()
        g = unique_glabels[ind]

        # Select corresponding probe samples
        ind = []
        for i, label in enumerate(plabels):
            if label in g: ind.append(i)
        ind = np.asarray(ind)

        p = plabels[ind]

        # Randomly select one sample per selected label
        subdist = np.zeros((n_selected_labels, p.size))
        for i, glabel in enumerate(g):
            samples = np.where(glabels == glabel)[0]
            j = np.random.choice(samples)
            for k in range(p.size):
                subdist[i, k] = distfunc(j, ind[k])

        # Compute CMC
        ret += _cmc_core(subdist, g, p)

    return ret / n_repeat


def compute_distmat(gallery, probe):
    """
    compute distance of each image feature pairs in gallery and probe, L2 norm
    :param gallery:     gallery image features, 2D ndarray(number of gallery x feature dim)
    :param probe:       probe image features, 2D ndarray(number of probe x feature dim)
    :return:            distance mat, 2D ndarray (number of gallery x number of probe)
    """
    return cdist(gallery, probe)


def sorted_image_names(distmat, g_name_array, top_n):
    """
    Based on distance mat, return sorted gallery image names.
    Input distmat is an ndarray(n_gallery x n_probe), result is (n_probe x top_n),
    each row represents a quary result of that probe.
    :param distmat:         distance mat, 2D ndarray (n_gallery x n_probe)
    :param g_name_array:    name of gallery images, must be a 1D ndarray (1 x n_gallery)
    :param top_n:           number of images that matches each probe image
    :return:                the top n images matches each probe image, 2D ndarray (n_probe x top_n)
    """
    distmat = np.transpose(distmat)
    m, n = distmat.shape
    if n > top_n:
        order = np.argsort(distmat, axis=1)[:, :top_n]
    else:
        order = np.argsort(distmat, axis=1)
    return g_name_array[order]


def _map_core(D, G, P, top_n):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)  # G[order],按距离排序的矩阵，第一行为top1,从左到右为各plabel对应的glabel
    cump = match.cumsum(0) * 1.0 / matlib.repmat(np.arange(1, m + 1, 1).reshape([m, 1]), 1, n)  # 从上至下累积求和
    p = (match * cump)[:top_n]
    ap = p.sum(axis=0) / match[:top_n].sum(0)
    ap[np.isnan(ap)] = 0
    return ap.mean()


def mAP(distmat, glabels=None, plabels=None, top_n=None, n_repeat=10):
    """Compute the Mean Average Precision(MAP)
    Args:
        distmat: A ``m×n`` distance matrix. ``m`` and ``n`` are the number of
            gallery and probe samples, respectively. In the case of ``glabels``
            and ``plabels`` both are ``None``, we assume both gallery and probe
            samples to be unique, i.e., the i-th gallery samples matches only to
            the i-th probe sample.
        glabels: Vector of length ``m`` that represents the labels of gallery
            samples
        plabels: Vector of length ``n`` that represents the labels of probe
            samples
        top_n: Select only part of all the labels to compute.
        n_repeat: The number of random sampling times
    Returns:
        A float number represents the MAP
    """
    m, n = distmat.shape
    if glabels is None and plabels is None:
        glabels = np.arange(0, m)
        plabels = np.arange(0, n)
    if type(glabels) is list:
        glabels = np.asarray(glabels)
    if type(plabels) is list:
        plabels = np.asarray(plabels)

    unique_glabels = np.unique(glabels)
    if top_n is None:
        top_n = unique_glabels.size
    ret1 = 0
    # for r in range(n_repeat):
    #     # Randomly select gallery labels
    #     ind = np.random.choice(unique_glabels.size,
    #                            top_n,
    #                            replace=False)
    #     ind.sort()
    #     g = unique_glabels[ind]
    #
    #     # Select corresponding probe samples
    #     ind = []
    #     for i, label in enumerate(plabels):
    #         if label in g: ind.append(i)
    #     ind = np.asarray(ind)
    #
    #     p = plabels[ind]
    #
    #     # Randomly select one sample per selected label
    #     subdist = np.zeros((top_n, p.size))
    #     for i, glabel in enumerate(g):
    #         samples = np.where(glabels == glabel)[0]
    #         j = np.random.choice(samples)
    #         subdist[i, :] = distmat[j, ind]
    #
    #     # Compute MAP
    #     ret1 += _map_core(subdist, g, p, top_n)
    ret2 = _map_core(distmat, glabels, plabels, top_n)

    return ret1 / n_repeat, ret2


def train_1000_mAP():
    print('train_1000_mAP()')
    # valid mAP
    g = np.load('VGG_model/result/test_features/train_1000_gallery_features.npy')
    g_labels = np.load('VGG_model/result/test_features/train_1000_gallery_labels.npy')
    p = np.load('VGG_model/result/test_features/train_1000_probe_features.npy')
    p_labels = np.load('VGG_model/result/test_features/train_1000_probe_labels.npy')
    distmat = compute_distmat(g, p)
    map1, map2 = mAP(distmat, glabels=g_labels, plabels=p_labels, top_n=200)
    print('train_1000 map: %f, %f ' % (map1, map2))


def valid_mAP():
    print('valid_mAP()')
    min_step = 10000000
    max_step = 0
    for root, dirs, files in os.walk(os.path.abspath('./VGG_model/result/test_features')):
        for name in files:
            if 'valid_probe_features_step-' in name:
                step = int(name.split('.')[0].split('-')[-1])
                if step > max_step:
                    max_step = step
                if step < min_step:
                    min_step = step
    map2_all = []
    # valid mAP
    for step in range(min_step, max_step + 1, 5000):
        g = np.load('VGG_model/result/test_features/valid_gallery_features_step-%d.npy' % step)
        g_labels = np.load('VGG_model/result/test_features/valid_gallery_labels_step-%d.npy' % step)
        p = np.load('VGG_model/result/test_features/valid_probe_features_step-%d.npy' % step)
        p_labels = np.load('VGG_model/result/test_features/valid_probe_labels_step-%d.npy' % step)
        distmat = compute_distmat(g, p)
        map1, map2 = mAP(distmat, glabels=g_labels, plabels=p_labels, top_n=200)
        map2_all.append(map2)
        print('step: %d, map: %f, %f ' % (step, map1, map2))
    plt.plot(range(min_step, max_step + 1, 5000), map2_all)
    plt.show()

def create_xml(pname, gnames):
    doc = Document()
    message = doc.createElement('Message')
    message.setAttribute('Version', '1.0')
    doc.appendChild(message)

    info = doc.createElement('Info')
    info.setAttribute('evaluateType', '11')
    info.setAttribute('mediaFile', 'PedestrianRetrieval')
    message.appendChild(info)

    items = doc.createElement('Items')
    message.appendChild(items)

    for i, item_name in enumerate(pname):
        item = doc.createElement('Item')
        item.setAttribute('imageName', str(item_name).zfill(6))
        gname_str = ''
        for gname in gnames[i]:
            gname_str += (str(gname).zfill(6) + ' ')
        gname_str = gname_str[:-1]
        item.appendChild(doc.createTextNode(gname_str))
        items.appendChild(item)

    fp = open('data/predict_result.xml', 'w')
    doc.writexml(fp, addindent='  ', newl='\n')


def generate_predict_xml():
    print('generate_predict_xml()')
    g = np.load('VGG_model/result/test_features/predict_gallery_features.npy')
    p = np.load('VGG_model/result/test_features/predict_probe_features.npy')
    g_label = np.load('VGG_model/result/test_features/predict_gallery_labels.npy')
    p_label = np.load('VGG_model/result/test_features/predict_probe_labels.npy')
    if False in (g_label == np.array(range(58061))):
        print('g_label order error')
        return
    if False in (p_label == np.array(range(4480))):
        print('p_label order error')
        return
    with open('data/predict_gallery_name.pkl', "rb") as f:
        g_names = pickle.load(f)
        print g_names[:10]
    with open('data/predict_probe_name.pkl', "rb") as f:
        p_names = pickle.load(f)
    print('start compute distance')
    distmat = compute_distmat(g, p)
    print('start sort')
    sort_g_names = sorted_image_names(distmat, g_names, top_n=200)
    print('start create xml')
    create_xml(p_names, sort_g_names)


if __name__ == '__main__':
    # train_1000_mAP()
    # valid_mAP()
    generate_predict_xml()
    # have to
    # 1. delete xml's first line(<?xml version="1.0"?>)
    # 2. delete last line(nothing in last line)
    # 3. add a space in the second line, after "PedestrianRetrieval"
    #    so the second line will be:<Info evaluateType="11" mediaFile="PedestrianRetrieval" />
    # 4. the xml's size is 6.4 MB (6,433,396 bytes)
