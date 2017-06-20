#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.spatial.distance import cdist


def _cmc_core(D, G, P):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)         # G[order],按距离排序的矩阵，第一行为top1,从左到右为各plabel对应的glabel
    return (match.sum(axis=1) * 1.0 / n).cumsum()   # 从上至下累积求和

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
    return cdist(gallery,probe)


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
    order = np.argsort(distmat, axis=1)
    if n > top_n:
        return g_name_array[order][:,:top_n]
    else:
        return g_name_array[order]

def _map_core(D, G, P, top_n):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)                 # G[order],按距离排序的矩阵，第一行为top1,从左到右为各plabel对应的glabel
    cump = match.cumsum(0)*1.0 / m         # 从上至下累积求和
    p = (match*cump)[:top_n]
    return np.mean(p.sum(axis=1))


def map(distmat,glabels=None, plabels=None, top_n=None, n_repeat=10):
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
        A vector represents the MAP
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
    ret2 = 0
    for r in range(n_repeat):
        # Randomly select gallery labels
        ind = np.random.choice(unique_glabels.size,
                               top_n,
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
        subdist = np.zeros((top_n, p.size))
        for i, glabel in enumerate(g):
            samples = np.where(glabels == glabel)[0]
            j = np.random.choice(samples)
            subdist[i, :] = distmat[j, ind]

        # Compute MAP
        ret1 += _map_core(subdist, g, p, top_n)
        ret2 += _map_core(distmat, glabels, plabels,top_n)

    return ret1 / n_repeat, ret2/ n_repeat

# a demo
if __name__ == '__main__':
    #计算距离矩阵 demo
    g = np.array([[0, 1],[1,5],[3,2],[3,3]])
    g_labels = [1,2,3,2]
    p = np.array([[1,0],[2,3],[1,1]])
    p_labels = [1,2,3]
    distmat = compute_distmat(g, p)
    g_names = np.sum(g**2,axis=1)
    # #计算cmc
    cmc_mean = count(distmat=distmat,n_selected_labels=3,n_repeat=10)
    map1, map2 = map(distmat,glabels=g_labels,plabels=p_labels)
    print('cmc ', cmc_mean,'map ',map1,map2)
    sort_g_names = sorted_image_names(distmat, g_names, top_n=2)
    print(sort_g_names)
    print('done')