#!/usr/bin/python2
# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import os, sys
import random
from sklearn.model_selection import train_test_split
from cmc import *


def get_dict_ids_images():
    vali_info = ElementTree.parse('data/PedestrianRetrieval_vali/vali.xml')

    root = vali_info.getroot()
    # print(root.tag)

    current_file_path = sys.path[0]
    # print(current_file_path)

    train_eval = {}

    for child in root:
        print(child.tag, child.attrib)
        if child.tag == 'Items':
            if child.attrib['name'] == 'ref':
                image_folder_path = os.path.join(current_file_path, 'data/PedestrianRetrieval_vali/vr_path')
            elif child.attrib['name'] == 'query':
                image_folder_path = os.path.join(current_file_path, 'data/PedestrianRetrieval_vali/vq_path')

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
    gallery_folder_path = os.path.join(current_file_path, 'data/new_online_vali_set_UPLOAD_VERSION/vq_path')
    prob_folder_path = os.path.join(current_file_path, 'data/new_online_vali_set_UPLOAD_VERSION/vr_path')
    gallery_predict = []
    prob_predict = []

    for root, dirs, files in os.walk(gallery_folder_path):
        for name in files:
            gallery_predict.append(os.path.join(gallery_folder_path, name))
    for root, dirs, files in os.walk(prob_folder_path):
        for name in files:
            prob_predict.append(os.path.join(prob_folder_path, name))
    return gallery_predict, prob_predict


def generate_gallery(img_dict, n):
    """
    generate gallery and probe
    gallery -> vq_path (id is unique in vq_path)
    probe -> vr_path
    :param image_dict:  id:image dictionary, a dict
    :param n:           size of gallery (n different ids), int
    :return:            [gallery(str list of image names), probe(str list of image names),
                        glabels(ndarray), plabels(ndarray)]
    """
    gallery = []
    probe = []
    glabels = []
    plabels = []
    for id in img_dict.keys():
        imgs = img_dict[id]
        img = random.choice(imgs)  # choice a image and put it into gallery, others in probe
        imgs.pop(imgs.index(img))
        gallery.append(img)
        glabels.append(int(id))
        for img in imgs:
            plabels.append(int(id))
            probe.append(img)
        if len(glabels) >= n:
            break

    return gallery, probe, np.array(glabels).astype(np.int32), np.array(plabels).astype(np.int32)


def get_neg_id(ref_id, info_dict):
    while (True):
        neg_id = random.sample(info_dict.keys(), 1)[0]
        if neg_id != ref_id:
            break
    return neg_id


def get_neg_image(ref_id, info_dict):
    neg_id = get_neg_id(ref_id, info_dict)
    neg_list = info_dict[neg_id]
    neg_image = random.sample(neg_list, 1)[0]
    return neg_image


def get_pos_image(ref_image, ref_id, info_dict):
    pos_list = info_dict[ref_id]
    pos_list.remove(ref_image)
    pos_image = random.sample(pos_list, 1)[0]
    pos_list.append(ref_image)
    return pos_image


'''
def get_animal_image():
    pos_list = info_dict[ref_id]
    pos_list.remove(ref_image)
    pos_image = random.sample(pos_list,1)[0]
    pos_list.append(ref_image)
    return pos_image
    
def get_triplet_pair_test():
    dataset_triplet_pair = []
    for key_id, value_images_list in train_eval.iteritems():
        #make every image in train_eval as reference image
        for ref_image in value_images_list:
            ref_pos_image = get_pos_image(ref_image, key_id, train_eval)
            ref_neg_image = get_animal_image()
            dataset_triplet_pair.append([ref_image, ref_pos_image, ref_neg_image])

    return dataset_triplet_pair
'''


def get_triplet_pair(train_eval):
    dataset_triplet_pair = []
    for key_id, value_images_list in train_eval.iteritems():
        # make every image in train_eval as reference image
        for ref_image in value_images_list:
            ref_pos_image = get_pos_image(ref_image, key_id, train_eval)
            ref_neg_image = get_neg_image(key_id, train_eval)
            dataset_triplet_pair.append([ref_image, ref_pos_image, ref_neg_image])

    return dataset_triplet_pair


def generate_train_eval(dataset_triplet_pair, csv_path):
    with open(csv_path, 'w') as output:
        for (ref_image, ref_pos_image, ref_neg_image) in dataset_triplet_pair:
            output.write("%s,%s,%s" % (ref_image, ref_pos_image, ref_neg_image))
            output.write("\n")


def generate_test(gallery_probe, glabels_plabels, test_batch):
    test_num = 0
    with open('data/test.csv', 'w') as output:
        for (image_path, image_label) in zip(gallery_probe, glabels_plabels):
            output.write("%s,%s" % (image_path, image_label))
            output.write("\n")

            test_num = test_num + 1

        while (test_num % test_batch != 0):
            output.write("%s,%s" % (image_path, 'Repeat for batch input'))
            output.write("\n")
            test_num = test_num + 1


def generate_path_csv(image_path_list, path_csv):
    with open(path_csv, 'w') as output:
        for image_path in image_path_list:
            output.write("%s" % (image_path))
            output.write("\n")


def generate_path_label_csv(path_csv, image_path_list, label = None):
    with open(path_csv, 'w') as output:
        if label is None:
            for i in range(len(image_path_list)):
                output.write("%s,%s" % (image_path_list[i], -1))
                output.write("\n")
        else:
            for i in range(len(image_path_list)):
                output.write("%s,%s" % (image_path_list[i], label[i]))
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


if __name__ == '__main__':
    # import train and valid data
    id_path = get_dict_ids_images()
    # split train and valid date
    id_path_array = dict2array(id_path)
    X_train_array, X_valid_array = train_test_split(id_path_array, test_size=0.15, random_state=1)
    X_train = array2dict(X_train_array)
    X_valid = array2dict(X_valid_array)
    print('train id: ' + str(len(X_train)))
    print('valid id: ' + str(len(X_valid)))
    # generate train csv
    dataset_triplet_pair = get_triplet_pair(X_train)
    generate_train_eval(dataset_triplet_pair, 'data/train.csv')
    print('train triplet_pair: ' + str(len(dataset_triplet_pair)))
    # generate valid csv
    gallery_valid, probe_valid, glabels_valid, plabels_valid = generate_gallery(X_valid, len(X_valid))
    # valid can be divided by base_number
    base_number = 10
    gallery_valid, probe_valid, glabels_valid, plabels_valid = map(lambda x: x[:len(x) - len(x) % base_number],
                                                                   [gallery_valid, probe_valid, glabels_valid,
                                                                    plabels_valid])
    generate_path_label_csv('data/valid_gallery.csv', gallery_valid, glabels_valid)
    generate_path_label_csv('data/valid_probe.csv', probe_valid, plabels_valid)
    print('gallery_valid: ' + str(len(gallery_valid)))
    print('probe_valid: ' + str(len(probe_valid)))
    # generate predict csv
    gallery_predict, prob_predict = generate_predict_path()
    generate_path_label_csv('data/predict_gallery.csv', gallery_predict)
    generate_path_label_csv('data/predict_probe.csv', prob_predict)
    print('gallery_predict: ' + str(len(gallery_predict)))
    print('prob_predict: ' + str(len(prob_predict)))
