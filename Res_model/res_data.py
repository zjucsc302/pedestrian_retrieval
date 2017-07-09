#!/usr/bin/python2
# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import os, sys
import random
from sklearn.model_selection import train_test_split
import cPickle as pickle
import numpy as np
import skimage.io
import skimage.transform
import math
from res_trainable import IMAGE_HEIGHT, IMAGE_WIDTH


def get_dict_ids_images():
    vali_info = ElementTree.parse(os.path.abspath('../data/PedestrianRetrieval_vali/vali.xml'))

    root = vali_info.getroot()
    # print(root.tag)

    train_eval = {}

    for child in root:
        print(child.tag, child.attrib)
        if child.tag == 'Items':
            if child.attrib['name'] == 'ref':
                image_folder_path = os.path.abspath('../data/PedestrianRetrieval_vali/vr_path')
            elif child.attrib['name'] == 'query':
                image_folder_path = os.path.abspath('../data/PedestrianRetrieval_vali/vq_path')

        for child_item in child.findall('Item'):
            child_item_pedestrianID = child_item.attrib['pedestrianID']
            child_item_imageName = child_item.attrib['imageName']

            child_item_image_path = os.path.join(image_folder_path, child_item_imageName)

            if child_item_pedestrianID not in train_eval.keys():
                train_eval[child_item_pedestrianID] = []
            train_eval[child_item_pedestrianID].append(child_item_image_path)

    # print train_eval['381126']
    return train_eval


def generate_predict_path():
    current_file_path = sys.path[0]
    prob_folder_path = os.path.join(current_file_path, 'data/new_online_vali_set_UPLOAD_VERSION/vq_path')
    gallery_folder_path = os.path.join(current_file_path, 'data/new_online_vali_set_UPLOAD_VERSION/vr_path')
    prob_predict = []
    gallery_predict = []
    for root, dirs, files in os.walk(prob_folder_path):
        for name in files:
            prob_predict.append(os.path.join(prob_folder_path, name))
    for root, dirs, files in os.walk(gallery_folder_path):
        for name in files:
            gallery_predict.append(os.path.join(gallery_folder_path, name))
    return gallery_predict, prob_predict


def generate_gallery(img_dict, n):
    """
    generate gallery and probe
    gallery -> vr_path (image number of a same id more than 3 in vq_path)
    probe -> vq_path (image number of a same id is 1 or 2 in vq_path)
    :param image_dict:  id:image dictionary, a dict
    :param n:           size of probe (n different ids), int
    :return:            [probe(str list of image names), gallery(str list of image names),
                        plabels(ndarray), glabels(ndarray)]
    """
    probe = []
    gallery = []
    plabels = []
    glabels = []
    for id in img_dict.keys()[:n]:
        imgs = img_dict[id]
        # choice a image and put it into probe
        img = random.choice(imgs)
        imgs.pop(imgs.index(img))
        probe.append(img)
        plabels.append(int(id))
        # 50% probability choice a image and put it into probe, others in gallery
        if random.random() > 0.5:
            img = random.choice(imgs)
            imgs.pop(imgs.index(img))
            probe.append(img)
            plabels.append(int(id))
        # generate gallery, max_num = 40
        for img in (imgs if len(imgs) < 41 else random.sample(imgs, 40)):
            gallery.append(img)
            glabels.append(int(id))
    return probe, gallery, np.array(plabels).astype(np.int32), np.array(glabels).astype(np.int32)


def generate_path_label_order_csv(path_csv, image_path_list, label=None):
    with open(path_csv, 'w') as output:
        if label is None:
            for i in range(len(image_path_list)):
                output.write("%s,%s,%s" % (image_path_list[i], -1, i))
                output.write("\n")
        else:
            for i in range(len(image_path_list)):
                output.write("%s,%s,%s" % (image_path_list[i], label[i], i))
                output.write("\n")


def generate_path_label_order_repeat_probe_csv(path_csv, repeat_num, image_path_list, label=None):
    with open(path_csv, 'w') as output:
        if label is None:
            for i in range(len(image_path_list)):
                for j in range(repeat_num):
                    output.write("%s,%s,%s" % (image_path_list[i], -1, i))
                    output.write("\n")
        else:
            for i in range(len(image_path_list)):
                for j in range(repeat_num):
                    output.write("%s,%s,%s" % (image_path_list[i], label[i], i))
                    output.write("\n")


def generate_name_order(path_csv, image_path_list):
    with open(path_csv, 'w') as output:
        for i in range(len(image_path_list)):
            name = image_path_list[i].split('/')[-1].split('.')[0]
            output.write("%s,%s" % (name, i))
            output.write("\n")


def dict2array(dict):
    array = []
    for key in dict:
        array.append([])
        array[-1].append(key)
        for value in dict[key]:
            array[-1].append(value)
    return array


def array2dict(array):
    dict = {}
    for item in array:
        dict[item[0]] = []
        dict[item[0]].append(item[1])
        for element in item[2:]:
            dict[item[0]].append(element)
    return dict


def get_id_path_dict(file_path):
    pfile = file_path
    if os.path.exists(pfile):
        print(pfile + ' exist, load from it')
        with open(pfile, "rb") as f:
            id_path = pickle.load(f)
    else:
        print('create id_path from vali.xml')
        id_path = get_dict_ids_images()
        with open(pfile, "wb") as f:
            pickle.dump(id_path, f)
    return id_path


def generate_id_image_file(id_path_dict, file_path, id_num_in_part):
    print('generate_id_image_file()')
    id_num = len([id for id in id_path_dict])
    part_num = id_num / id_num_in_part
    print('id_num: %s' % id_num)
    print('id_num_in_part: %s' % id_num_in_part)
    print('part_num: %s' % part_num)

    id_image_dict = {}
    id_count = 0
    id_part_count = 0
    image_count = 0

    for id in id_path_dict:
        images = []
        for path in id_path_dict[id]:
            image = skimage.io.imread(path)
            image = skimage.transform.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH)) * 255.0
            image = image.astype(np.uint8)
            # print image.dtype
            # print image.shape
            # print(image)
            # skimage.io.imshow(image)
            # skimage.io.show()
            images.append(image)
            # image_count += 1
            # print image_count
        id_image_dict[id] = images

        id_count += 1
        if id_count == id_num_in_part:
            pfile = file_path % (IMAGE_HEIGHT, IMAGE_WIDTH, id_part_count)
            print('generate ' + pfile)
            with open(pfile, "wb") as f:
                pickle.dump(id_image_dict, f)
            id_count = 0
            id_image_dict = {}
            id_part_count += 1


def generate_image_file(image_path_list, file_path, image_num_in_part):
    print('generate_image_file()')
    image_num = len(image_path_list)
    print('image num: %s' % image_num)
    print('image num in part: %s' % image_num_in_part)
    part_num = int(math.ceil(1.0 * image_num / image_num_in_part))
    print('part num: %s' % part_num)

    images = []
    path_count = 0
    for i, path in enumerate(image_path_list):
        image = skimage.io.imread(path)
        image = skimage.transform.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH)) * 255.0
        image = image.astype(np.uint8)
        images.append(image)
        if (i + 1) % image_num_in_part == 0:
            pfile = file_path % (IMAGE_HEIGHT, IMAGE_WIDTH, path_count)
            print('generate ' + pfile)
            with open(pfile, "wb") as f:
                pickle.dump(images, f)
            images = []
            path_count += 1


def generate_label_file(nplabels, file_path):
    np.save(os.path.abspath(file_path), nplabels)


def generate_path_label():
    print('generate_path_label()')
    # import train and valid path
    id_path = get_id_path_dict('../data/id_path.pkl')
    # split train and valid path
    id_path_array = dict2array(id_path)
    X_train_array, X_valid_array = train_test_split(id_path_array, test_size=0.04, random_state=1)
    X_train = array2dict(X_train_array)
    X_valid = array2dict(X_valid_array)
    print('train id: ' + str(len(X_train)))
    print('valid id: ' + str(len(X_valid)))
    # generate train file
    generate_id_image_file(X_train, '../data/id_image/id_image_train_%s_%s_%s.pkl', id_num_in_part=100)
    # generate train_200 file
    probe_train, gallery_train, plabels_train, glabels_train = generate_gallery(X_train, 200)
    generate_label_file(plabels_train, './result/test_features/train_200_probe_labels.npy')
    generate_label_file(glabels_train, './result/test_features/train_200_gallery_labels.npy')
    generate_image_file(probe_train, '../data/id_image/train_200_probe_image_%s_%s_%s.pkl', len(probe_train))
    generate_image_file(gallery_train, '../data/id_image/train_200_gallery_image_%s_%s_%s.pkl', len(gallery_train))
    generate_path_label_order_csv('../data/train_200_probe.csv', probe_train, plabels_train)  # just for count
    generate_path_label_order_csv('../data/train_200_gallery.csv', gallery_train, glabels_train)  # just for count
    print('probe_train_200: ' + str(len(probe_train)))
    print('gallery_train_200: ' + str(len(gallery_train)))
    # generate valid file
    probe_valid, gallery_valid, plabels_valid, glabels_valid = generate_gallery(X_valid, len(X_valid))
    generate_label_file(plabels_valid, './result/test_features/valid_probe_labels.npy')
    generate_label_file(glabels_valid, './result/test_features/valid_gallery_labels.npy')
    generate_image_file(probe_valid, '../data/id_image/valid_probe_image_%s_%s_%s.pkl', len(probe_valid))
    generate_image_file(gallery_valid, '../data/id_image/valid_gallery_image_%s_%s_%s.pkl', len(gallery_valid))
    generate_path_label_order_csv('../data/valid_probe.csv', probe_valid, plabels_valid)  # just for count
    generate_path_label_order_csv('../data/valid_gallery.csv', gallery_valid, glabels_valid)  # just for count
    print('probe_valid: ' + str(len(probe_valid)))
    print('gallery_valid: ' + str(len(gallery_valid)))
    # generate predict csv
    gallery_predict, probe_predict = generate_predict_path()
    generate_image_file(probe_predict, '../data/id_image/predict_probe_image_%s_%s_%s.pkl', 2000)
    generate_image_file(gallery_predict, '../data/id_image/predict_gallery_image_%s_%s_%s.pkl', 2000)
    generate_path_label_order_csv('data/predict_probe.csv', probe_predict)
    generate_path_label_order_csv('data/predict_gallery.csv', gallery_predict)
    print('prob_predict: ' + str(len(probe_predict)))
    print('gallery_predict: ' + str(len(gallery_predict)))
    # generate predict name csv
    generate_name_order('data/predict_probe_name.csv', probe_predict)
    generate_name_order('data/predict_gallery_name.csv', gallery_predict)


if __name__ == '__main__':
    generate_path_label()
