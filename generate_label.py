#!/usr/bin/python2
# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import os, sys
import random
from sklearn.model_selection import train_test_split
from cmc import *
import cPickle as pickle


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
        # generate gallery
        for img in imgs:
            gallery.append(img)
            glabels.append(int(id))
    return probe, gallery, np.array(plabels).astype(np.int32), np.array(glabels).astype(np.int32)


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
    order = 0
    with open(csv_path, 'w') as output:
        for (ref_image, ref_pos_image, ref_neg_image) in dataset_triplet_pair:
            output.write("%s,%s,%s,%s" % (ref_image, ref_pos_image, ref_neg_image, order))
            output.write("\n")
            order += 1


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


def generate_name_np(path_np, image_path_list):
    name_np = np.zeros(len(image_path_list), dtype=np.int32)
    for i in range(len(image_path_list)):
        name_np[i] = (image_path_list[i].split('/')[-1].split('.')[0])
    with open(path_np, "wb") as f:
        pickle.dump(name_np, f)


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


def generate_path_label():
    print('generate_path_label()')
    # import train and valid data
    pfile = 'data/id_path.pkl'
    if os.path.exists(pfile):
        print('id_path.pkl exist, load from id_path.pkl')
        with open(pfile, "rb") as f:
            id_path = pickle.load(f)
    else:
        print('create id_path from vali.xml')
        id_path = get_dict_ids_images()
        with open(pfile, "wb") as f:
            pickle.dump(id_path, f)
    # split train and valid date
    id_path_array = dict2array(id_path)
    X_train_array, X_valid_array = train_test_split(id_path_array, test_size=0.15, random_state=1)
    X_train = array2dict(X_train_array)
    X_valid = array2dict(X_valid_array)
    print('train id: ' + str(len(X_train)))
    print('valid id: ' + str(len(X_valid)))
    # generate train_triplet_pair csv
    dataset_triplet_pair = get_triplet_pair(X_train)
    generate_train_eval(dataset_triplet_pair, 'data/train_triplet_pair.csv')
    print('train triplet_pair: ' + str(len(dataset_triplet_pair)))
    # generate train_1000 csv
    probe_train, gallery_train, plabels_train, glabels_train = generate_gallery(X_train, 1000)
    generate_path_label_order_csv('data/train_1000_probe.csv', probe_train, plabels_train)
    generate_path_label_order_csv('data/train_1000_gallery.csv', gallery_train, glabels_train)
    print('probe_train_1000: ' + str(len(probe_train)))
    print('gallery_train_1000: ' + str(len(gallery_train)))
    # generate valid csv
    probe_valid, gallery_valid, plabels_valid, glabels_valid = generate_gallery(X_valid, len(X_valid))
    generate_path_label_order_csv('data/valid_probe.csv', probe_valid, plabels_valid)
    generate_path_label_order_csv('data/valid_gallery.csv', gallery_valid, glabels_valid)
    print('probe_valid: ' + str(len(probe_valid)))
    print('gallery_valid: ' + str(len(gallery_valid)))
    # generate predict csv
    gallery_predict, probe_predict = generate_predict_path()
    generate_path_label_order_csv('data/predict_probe.csv', probe_predict)
    generate_path_label_order_csv('data/predict_gallery.csv', gallery_predict)
    print('prob_predict: ' + str(len(probe_predict)))
    print('gallery_predict: ' + str(len(gallery_predict)))
    # generate predict name csv
    generate_name_order('data/predict_probe_name.csv', probe_predict)
    generate_name_order('data/predict_gallery_name.csv', gallery_predict)


if __name__ == '__main__':
    generate_path_label()
