

from xml.etree import ElementTree

import pickle as pkl
import os,sys
import random
from sklearn.model_selection import train_test_split


def get_dict_ids_images():

    vali_info = ElementTree.parse('data/PedestrianRetrieval_vali/vali.xml')

    root = vali_info.getroot()
    #print(root.tag)

    current_file_path = sys.path[0]
    #print(current_file_path)

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

    #print train_eval['381126']
    return train_eval

def get_neg_id(ref_id, info_dict):
    while(True):
        neg_id = random.sample(info_dict.keys(),1)[0]
        if neg_id != ref_id:
            break
    return neg_id

def get_neg_image(ref_id, info_dict):
    neg_id = get_neg_id(ref_id, info_dict)
    neg_list = info_dict[neg_id]
    neg_image = random.sample(neg_list,1)[0]
    return neg_image

def get_pos_image(ref_image, ref_id, info_dict):
    pos_list = info_dict[ref_id]
    pos_list.remove(ref_image)
    pos_image = random.sample(pos_list,1)[0]
    pos_list.append(ref_image)
    return pos_image

def get_triplet_pair(train_eval):
    dataset_triplet_pair = []
    for key_id, value_images_list in train_eval.iteritems():
        #make every image in train_eval as reference image
        for ref_image in value_images_list:
            ref_pos_image = get_pos_image(ref_image, key_id, train_eval)
            ref_neg_image = get_neg_image(key_id, train_eval)
            dataset_triplet_pair.append([ref_image, ref_pos_image, ref_neg_image])

    return dataset_triplet_pair


def generate_train_eval(dataset_triplet_pair):
    X_train_val, X_test = train_test_split(dataset_triplet_pair, test_size=0.2, random_state=1)
    X_train, X_val = train_test_split(X_train_val, test_size=0.25, random_state=1)

    with open('data/train.csv', 'w') as output:
        for (ref_image, ref_pos_image, ref_neg_image) in X_train:
            output.write("%s,%s,%s" % (ref_image, ref_pos_image, ref_neg_image))
            output.write("\n")

    with open('data/test.csv', 'w') as output:
        for (ref_image, ref_pos_image, ref_neg_image) in X_test:
            output.write("%s,%s,%s" % (ref_image, ref_pos_image, ref_neg_image))
            output.write("\n")

    with open('data/eval.csv', 'w') as output:
        for (ref_image, ref_pos_image, ref_neg_image) in X_val:
            output.write("%s,%s,%s" % (ref_image, ref_pos_image, ref_neg_image))
            output.write("\n")



train_eval = get_dict_ids_images()

dataset_triplet_pair = get_triplet_pair(train_eval)

generate_train_eval(dataset_triplet_pair)




'''


'''

