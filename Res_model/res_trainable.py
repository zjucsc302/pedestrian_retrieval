IMAGE_HEIGHT = 224
IMAGE_WIDTH = 112


import tensorflow as tf
import numpy as np
import os
from tensorflow.python.platform import gfile
import csv
from vgg_preprocessing import my_preprocess_train
import random
import cPickle as pickle
from res_data import process_image
import skimage.io


class Train_Flags():
    def __init__(self):

        self.current_file_path = os.path.abspath('.')

        self.id_image_path = os.path.abspath('../data/id_image')
        self.id_path_train_path = os.path.abspath('../data/id_path_train.pkl')
        self.dataset_train_csv_file_path = os.path.abspath('../data/train_triplet_pair.csv')
        self.dataset_train_200_gallery_csv_file_path = os.path.abspath('../data/train_200_gallery.csv')
        self.dataset_train_200_probe_csv_file_path = os.path.abspath('../data/train_200_probe.csv')
        self.dataset_valid_gallery_csv_file_path = os.path.abspath('../data/valid_gallery.csv')
        self.dataset_valid_probe_csv_file_path = os.path.abspath('../data/valid_probe.csv')
        self.dataset_predict_gallery_csv_file_path = os.path.abspath('../data/predict_gallery.csv')
        self.dataset_predict_probe_csv_file_path = os.path.abspath('../data/predict_probe.csv')
        self.output_summary_path = os.path.join(self.current_file_path, 'result', 'summary')
        self.output_check_point_path = os.path.join(self.current_file_path, 'result', 'check_point')
        self.output_test_features_path = os.path.join(self.current_file_path, 'result', 'test_features')
        self.resnet_checkpoint_file = 'resnet_v2_50.ckpt'
        self.check_path_exist()
        self.checkpoint_name = 'resnet_finetuning.ckpt'

        self.max_step = 30001
        self.test_batch_size = 80  # do not change 80!!!
        self.change_file_step = 400

        self.initial_learning_rate = 0.0001
        self.decay_rate = 0.5
        self.decay_steps = 10000

        self.return_id_num = 20
        self.image_num_every_id = 4
        self.train_batch_size = self.return_id_num * self.image_num_every_id
        self.output_feature_dim = 128
        self.dropout = 0.9
        # self.tau1 = 0.6
        # self.tau2 = 0.01
        # self.beta = 0.002
        self.m = 0.4

        with open(self.dataset_train_200_gallery_csv_file_path, 'rb') as f:
            self.train_200_gallery_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_train_200_probe_csv_file_path, 'rb') as f:
            self.train_200_probe_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_valid_gallery_csv_file_path, 'rb') as f:
            self.valid_gallery_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_valid_probe_csv_file_path, 'rb') as f:
            self.valid_probe_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_predict_gallery_csv_file_path, 'rb') as f:
            self.predict_gallery_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_predict_probe_csv_file_path, 'rb') as f:
            self.predict_probe_num = sum([1 for row in csv.reader(f)])
        for root, dirs, files in os.walk(self.id_image_path):
            self.id_image_train_num = sum([1 if 'id_image_train' in file_name else 0 for file_name in files])

    def check_path_exist(self):
        if not gfile.Exists(self.output_summary_path):
            gfile.MakeDirs(self.output_summary_path)
        if not gfile.Exists(self.output_check_point_path):
            gfile.MakeDirs(self.output_check_point_path)
        if not gfile.Exists(self.output_test_features_path):
            gfile.MakeDirs(self.output_test_features_path)


class ResnetReid:
    """
    A trainable version VGG19.
    """

    def __init__(self, dropout=0.5):
        self.dropout = dropout

    def process_image(self, image_batch, image_num):
        '''image_batch:[batch, height, width, 3], tf.float32, [0.0, 1.0]'''
        images_process_list = []
        images = tf.split(image_batch, image_num)
        for image in images:
            image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            image = my_preprocess_train(image, IMAGE_HEIGHT, IMAGE_WIDTH)
            image = tf.reshape(image, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            images_process_list.append(image)
        return tf.concat(images_process_list, 0)

    def build(self, resnet_avg_pool, train_test_mode):
        """
        :param train_test_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        assert resnet_avg_pool.get_shape().as_list()[1:] == [1, 1, 2048]
        self.resnet_avg_pool_flat = tf.reshape(resnet_avg_pool, [-1, 1 * 1 * 2048])
        self.fc1_add = tf.layers.dense(inputs=self.resnet_avg_pool_flat, units=1024,
                                       kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01), use_bias=False,
                                       name='fc1_add')
        self.bn1_add = tf.layers.batch_normalization(inputs=self.fc1_add, training=train_test_mode, name='bn1_add')
        assert self.bn1_add.get_shape().as_list()[1:] == [1024]
        self.relu1_add = tf.nn.relu(features=self.bn1_add)
        self.fc2_add = tf.layers.dense(inputs=self.relu1_add, units=128,
                                       kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1), use_bias=False,
                                       name='fc2_add')
        self.output = self.fc2_add

    # hard triplet loss
    def calc_loss(self, features, return_id_num, image_num_every_id, m):
        features = tf.cast(features, dtype=tf.float32)
        # features = tf.nn.l2_normalize(features, dim=1)

        # set distmat mask
        distmat_len = return_id_num * image_num_every_id
        mask_intra_np = np.zeros([distmat_len, distmat_len], dtype=np.float32)
        part = np.ones([image_num_every_id, image_num_every_id], dtype=np.float32)
        for i in range(return_id_num):
            mask_intra_np[i * image_num_every_id:(i + 1) * image_num_every_id,
            i * image_num_every_id:(i + 1) * image_num_every_id] = part
        mask_inter_np = 1 - mask_intra_np
        mask_intra = tf.constant(mask_intra_np, dtype=tf.float32)
        mask_inter = tf.constant(mask_inter_np, dtype=tf.float32)

        # calculate distmat
        M1 = distmat_len
        M2 = M1
        A = features
        B = features
        p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), tf.ones(shape=(1, M2)))
        p2 = tf.transpose(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]), tf.ones(shape=(M1, 1)),
                                    transpose_b=True))
        distmat_square = tf.abs(tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True))
        distmat_square = distmat_square + tf.cast(distmat_square < 1e-12, dtype=tf.float32) * 1e-12
        distmat = tf.sqrt(distmat_square)

        # calculate loss
        distmat_intra = mask_intra * distmat
        distmat_intra_max = tf.reduce_max(distmat_intra, axis=1, keep_dims=True)
        distmat_inter = mask_inter * distmat + mask_intra * tf.reduce_max(distmat)
        distmat_inter_min = tf.reduce_min(distmat_inter, axis=1, keep_dims=True)
        # losses = tf.maximum(m + distmat_intra_max - distmat_inter_min, 0) # hinge
        losses = tf.log(1 + tf.exp(distmat_intra_max - distmat_inter_min))  # soft-margin

        loss_mean = tf.reduce_mean(losses) # batch hard example
        loss_max = tf.reduce_max(losses) # batch hardest example
        tf.add_to_collection('loss', loss_mean)
        tf.summary.scalar('loss_mean', loss_mean)
        tf.summary.scalar('loss_max', loss_max)

        # calculate state
        distmat_intra_mean = tf.reduce_sum(distmat_intra) / tf.reduce_sum(mask_intra)
        distmat_intra_max = tf.reduce_max(distmat_intra_max)
        distmat_inter_mean = tf.reduce_sum(mask_inter * distmat) / tf.reduce_sum(mask_inter)
        distmat_inter_min = tf.reduce_min(distmat_inter_min)
        tf.summary.scalar('distmat_intra_mean', distmat_intra_mean)
        tf.summary.scalar('distmat_intra_max', distmat_intra_max)
        tf.summary.scalar('distmat_inter_mean', distmat_inter_mean)
        tf.summary.scalar('distmat_inter_min', distmat_inter_min)
        part00 = tf.slice(distmat, [0, 0], [image_num_every_id, image_num_every_id])
        part01 = tf.slice(distmat, [0, image_num_every_id], [image_num_every_id, image_num_every_id])
        part10 = tf.slice(distmat, [image_num_every_id, 0], [image_num_every_id, image_num_every_id])
        part11 = tf.slice(distmat, [image_num_every_id, image_num_every_id], [image_num_every_id, image_num_every_id])
        accuracy0 = tf.reduce_mean(tf.cast(part00 < part01, dtype=tf.float32))
        accuracy1 = tf.reduce_mean(tf.cast(part11 < part10, dtype=tf.float32))
        tf.summary.scalar('accuracy', (accuracy0 + accuracy1) / 2)

    # # improved max triplet loss
    # def calc_loss(self, logits, tau1, tau2, beta):
    #     # # for the reason of tf.nn.l2_normalize, input has the same dtype with the output, should be float
    #     logits = tf.cast(logits, dtype=tf.float32)
    #     logits = tf.nn.l2_normalize(logits, dim=1)
    #
    #     split_refs, split_poss, split_negs = tf.split(logits, num_or_size_splits=3, axis=0)
    #
    #     dist_ref_to_pos = tf.norm(split_refs - split_poss, 2, 1)
    #     dist_ref_to_neg = tf.norm(split_refs - split_negs, 2, 1)
    #     dist_pos_to_neg = tf.norm(split_poss - split_negs, 2, 1)
    #
    #     max_dis_in_triplet = tf.maximum(dist_ref_to_pos ** 2 - dist_ref_to_neg ** 2,
    #                                     dist_ref_to_pos ** 2 - dist_pos_to_neg ** 2)
    #     inter_const = tf.maximum(max_dis_in_triplet + tau1, 0.0)
    #     intra_const = tf.maximum(dist_ref_to_pos ** 2 - tau2, 0.0)
    #     costs = inter_const + beta * intra_const
    #     tf.add_to_collection('losses', costs)
    #
    #     tf.summary.scalar('inter_const_mean', tf.reduce_mean(dist_ref_to_neg))
    #     tf.summary.scalar('intra_const_mean', tf.reduce_mean(dist_ref_to_pos))
    #     tf.summary.scalar('inter_const_min', tf.reduce_min(dist_ref_to_neg))
    #     tf.summary.scalar('intra_const_max', tf.reduce_max(dist_ref_to_pos))
    #     accuracy = tf.reduce_mean(tf.cast(dist_ref_to_pos < dist_ref_to_neg, dtype=tf.float32))
    #     tf.summary.scalar('accuracy', accuracy)

    # # improved triplet loss
    # def calc_loss(self, logits, tau1, tau2, beta):
    #     # # for the reason of tf.nn.l2_normalize, input has the same dtype with the output, should be float
    #     logits = tf.cast(logits, dtype=tf.float32)
    #     logits = tf.nn.l2_normalize(logits, dim=1)
    #
    #     split_refs, split_poss, split_negs = tf.split(logits, num_or_size_splits=3, axis=0)
    #
    #     dist_ref_to_pos = tf.norm(split_refs - split_poss, 2, 1)
    #     dist_ref_to_neg = tf.norm(split_refs - split_negs, 2, 1)
    #
    #     inter_const = tf.maximum(dist_ref_to_pos - dist_ref_to_neg + tau1, 0.0)
    #     intra_const = tf.maximum(dist_ref_to_pos - tau2, 0.0)
    #     costs = inter_const + beta * intra_const
    #     tf.add_to_collection('losses', costs)
    #
    #     tf.summary.scalar('inter_const_mean', tf.reduce_mean(dist_ref_to_neg))
    #     tf.summary.scalar('intra_const_mean', tf.reduce_mean(dist_ref_to_pos))
    #     tf.summary.scalar('inter_const_max', tf.reduce_max(dist_ref_to_neg))
    #     tf.summary.scalar('intra_const_max', tf.reduce_max(dist_ref_to_pos))
    #     accuracy = tf.reduce_mean(tf.cast(dist_ref_to_pos < dist_ref_to_neg, dtype=tf.float32))
    #     tf.summary.scalar('accuracy', accuracy)

    def get_train_image_batch_direct(self, file_path, return_id_num, image_num_every_id):
        try:
            self.id_path is None
        except:
            with open(file_path, "rb") as f:
                self.id_path = pickle.load(f)
            self.ids = [i for i in self.id_path]
            print('load id_path')

        ids_select = random.sample(self.ids, return_id_num)
        images = np.zeros([return_id_num * image_num_every_id, IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=np.uint8)
        images_input_index = 0
        for id_select in ids_select:
            for path_select in random.sample(self.id_path[id_select], image_num_every_id):
                image = skimage.io.imread(path_select)
                image = process_image(image)
                # print image.dtype
                # print image.shape
                # print(image)
                # skimage.io.imshow(image)
                # skimage.io.show()
                images[images_input_index:images_input_index + 1, :, :, :] = image
                images_input_index += 1
        images = images.astype(np.float32) / 255
        # images: [0.0 1.0]
        return images

    def get_train_image_batch(self, folder_path, file_num, return_id_num, image_num_every_id, change_file=False):
        try:
            self.id_image_0 is None
        except:
            print('train image first load')
            change_file = True
        if change_file:
            two_file_number = random.sample(range(file_num), 2)
            file_path_0 = os.path.join(folder_path,
                                       'id_image_train_%s_%s_%s.pkl' % (IMAGE_HEIGHT, IMAGE_WIDTH, two_file_number[0]))
            file_path_1 = os.path.join(folder_path,
                                       'id_image_train_%s_%s_%s.pkl' % (IMAGE_HEIGHT, IMAGE_WIDTH, two_file_number[1]))
            with open(file_path_0, "rb") as f:
                self.id_image_0 = pickle.load(f)
            with open(file_path_1, "rb") as f:
                self.id_image_1 = pickle.load(f)
            print('change file to %s and %s' % (file_path_0, file_path_1))
            self.ids_0 = [i for i in self.id_image_0]
            self.ids_1 = [i for i in self.id_image_1]

        ids_part_0 = random.sample(self.ids_0, return_id_num / 2)
        ids_part_1 = random.sample(self.ids_1, return_id_num - return_id_num / 2)
        images = np.zeros([return_id_num * image_num_every_id, IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=np.uint8)
        images_input_index = 0

        for id_0 in ids_part_0:
            images[images_input_index:images_input_index + image_num_every_id, :, :, :] = np.array(
                random.sample(self.id_image_0[id_0], image_num_every_id))
            images_input_index += image_num_every_id
        for id_1 in ids_part_1:
            images[images_input_index:images_input_index + image_num_every_id, :, :, :] = np.array(
                random.sample(self.id_image_1[id_1], image_num_every_id))
            images_input_index += image_num_every_id
        images = images.astype(np.float32) / 255
        # images: [0.0 1.0]
        return images

    def get_test_image_batch(self, batch_index, batch_size, folder_path, file_title, gallery_flag):
        if gallery_flag:
            try:
                self.test_gallery_image is None
            except:
                print(file_title + ' gallery image first load')
                file_path = os.path.join(folder_path,
                                         file_title + '_gallery_image_%s_%s_0.pkl' % (IMAGE_HEIGHT, IMAGE_WIDTH))
                with open(file_path, "rb") as f:
                    self.test_gallery_image = pickle.load(f)
                    for i in range(batch_size):
                        self.test_gallery_image.append(self.test_gallery_image[0])

            images = np.array(self.test_gallery_image[batch_index: batch_index + batch_size], dtype=np.uint8)
            images = images.astype(np.float32) / 255
            # images: [0.0 1.0]
            return images
        else:
            try:
                self.test_probe_image is None
            except:
                print(file_title + ' probe image first load')
                file_path = os.path.join(folder_path,
                                         file_title + '_probe_image_%s_%s_0.pkl' % (IMAGE_HEIGHT, IMAGE_WIDTH))
                with open(file_path, "rb") as f:
                    self.test_probe_image = pickle.load(f)
                    for i in range(batch_size):
                        self.test_probe_image.append(self.test_probe_image[0])

            images = np.array(self.test_probe_image[batch_index: batch_index + batch_size], dtype=np.uint8)
            images = images.astype(np.float32) / 255
            # images: [0.0 1.0]
            return images

    def get_predict_image_batch(self, batch_index, batch_size, folder_path, file_title, gallery_flag):
        if gallery_flag:
            try:
                self.predict_gallery_image is None
            except:
                print(file_title + ' gallery image first load')
                self.file_count = 0
                for root, dirs, files in os.walk(folder_path):
                    self.file_num = sum([1 if 'predict_gallery_image' in file_name else 0 for file_name in files])
                file_path = os.path.join(folder_path, file_title + '_gallery_image_%s_%s_%s.pkl' % (
                    IMAGE_HEIGHT, IMAGE_WIDTH, self.file_count))
                with open(file_path, "rb") as f:
                    self.predict_gallery_image = pickle.load(f)
                print('load: %s' % file_path)
                self.image_num_in_part = len(self.predict_gallery_image)
                for i in range(batch_size):
                    self.predict_gallery_image.append(self.predict_gallery_image[0])
                self.batch_num_in_part = self.image_num_in_part / batch_size
                if self.image_num_in_part % batch_size != 0:
                    print('image_num_in_part can not divide by batch_size!')
                    print('image_num_in_part: %s, batch_size: %s' % (self.image_num_in_part, batch_size))
                    return
            if batch_index % self.image_num_in_part == 0 and batch_index != 0:
                self.file_count += 1
                file_path = os.path.join(folder_path, file_title + '_gallery_image_%s_%s_%s.pkl' % (
                    IMAGE_HEIGHT, IMAGE_WIDTH, self.file_count))
                with open(file_path, "rb") as f:
                    self.predict_gallery_image = pickle.load(f)
                print('load: %s' % file_path)
                for i in range(batch_size):
                    self.predict_gallery_image.append(self.predict_gallery_image[0])

            batch_index = batch_index - self.file_count * self.image_num_in_part
            images = np.array(self.predict_gallery_image[batch_index: batch_index + batch_size], dtype=np.uint8)
            print('batch_index_in_part: %s' % batch_index)
            images = images.astype(np.float32) / 255
            # images: [0.0 1.0]
            return images
        else:
            try:
                self.predict_probe_image is None
            except:
                print(file_title + ' probe image first load')
                self.file_count = 0
                for root, dirs, files in os.walk(folder_path):
                    self.file_num = sum([1 if 'predict_probe_image' in file_name else 0 for file_name in files])
                file_path = os.path.join(folder_path, file_title + '_probe_image_%s_%s_%s.pkl' % (
                    IMAGE_HEIGHT, IMAGE_WIDTH, self.file_count))
                with open(file_path, "rb") as f:
                    self.predict_probe_image = pickle.load(f)
                print('load: %s' % file_path)
                self.image_num_in_part = len(self.predict_probe_image)
                for i in range(batch_size):
                    self.predict_probe_image.append(self.predict_probe_image[0])
                self.batch_num_in_part = self.image_num_in_part / batch_size
                if self.image_num_in_part % batch_size != 0:
                    print('image_num_in_part can not divide by batch_size!')
                    print('image_num_in_part: %s, batch_size: %s' % (self.image_num_in_part, batch_size))
                    return
            if batch_index % self.image_num_in_part == 0 and batch_index != 0:
                self.file_count += 1
                file_path = os.path.join(folder_path, file_title + '_probe_image_%s_%s_%s.pkl' % (
                    IMAGE_HEIGHT, IMAGE_WIDTH, self.file_count))
                with open(file_path, "rb") as f:
                    self.predict_probe_image = pickle.load(f)
                print('load: %s' % file_path)
                for i in range(batch_size):
                    self.predict_probe_image.append(self.predict_probe_image[0])

            batch_index = batch_index - self.file_count * self.image_num_in_part
            images = np.array(self.predict_probe_image[batch_index: batch_index + batch_size], dtype=np.uint8)
            print('batch_index_in_part: %s' % batch_index)
            images = images.astype(np.float32) / 255
            # images: [0.0 1.0]
            return images
