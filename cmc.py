#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.spatial.distance import pdist, squareform,cdist
#from generate_label import get_dict_ids_images


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


def generate_gallery(img_dict, n):
    """
    generate gallery and probe
    :param image_dict:  id:image dictionary, a dict
    :param n:           size of gallery (n different ids), int
    :return:            [gallery(str list of image names), probe(str list of image names),
                        glabels（ndarray), plabels(ndarray)]
    """
    gallery = []
    probe = []
    glabels = []
    plabels = []
    for id in img_dict.keys():
        imgs = img_dict[id]
        img = random.choice(imgs)       # 对每个id,随机选一张图像放入gallery,剩下放probe
        imgs.pop(imgs.index(img))
        gallery.append(img)
        glabels.append(int(id))
        for img in imgs:
            plabels.append(int(id))
            probe.append(img)
        if len(glabels) >= n:
            break

    return gallery, probe, np.array(glabels).astype(np.int32), np.array(plabels).astype(np.int32)


def compute_distmat(gallery, probe):
    """
    compute distance of each image feature pairs in gallery and probe, L2
    :param gallery:     gallery image features, 2D ndarray(number of gallery x feature dim)
    :param probe:       probe image features, 2D ndarray(number of probe x feature dim)
    :return:            distance mat, 2D ndarray(number of gallery x number of probe)
    """
    return cdist(gallery, probe)



# 生成gallery和probe
#train_eval = get_dict_ids_images()
#gallery, probe, glabels, plabels = generate_gallery(train_eval, 800)

#计算距离矩阵 demo
#distmat = compute_distmat(np.array([[0, 1],[1,5],[3,2],[3,3]]),np.array([[1,0],[2,3],[1,1]]))
#计算cmc
#cmc_mean = count(distmat=distmat,glabels=glabels,plabels=plabels,n_selected_labels=50,n_repeat=10)
#print('done')