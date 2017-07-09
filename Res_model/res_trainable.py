import tensorflow as tf
import numpy as np
import os
from tensorflow.python.platform import gfile
import csv
# from pedestrian_retrieval.VGG_model.vgg_preprocessing import my_preprocess_train
import math
import random
import cPickle as pickle

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 112


class Train_Flags():
    def __init__(self):

        self.current_file_path = os.path.abspath('.')

        self.id_image_path = os.path.abspath('../data/id_image')
        self.dataset_train_csv_file_path = os.path.abspath('../data/train_triplet_pair.csv')
        self.dataset_train_1000_gallery_csv_file_path = os.path.abspath('../data/train_1000_gallery.csv')
        self.dataset_train_1000_probe_csv_file_path = os.path.abspath('../data/train_1000_probe.csv')
        self.dataset_valid_gallery_csv_file_path = os.path.abspath('../data/valid_gallery.csv')
        self.dataset_valid_probe_csv_file_path = os.path.abspath('../data/valid_probe.csv')
        self.dataset_predict_gallery_csv_file_path = os.path.abspath('../data/predict_gallery.csv')
        self.dataset_predict_probe_csv_file_path = os.path.abspath('../data/predict_probe.csv')
        self.output_summary_path = os.path.join(self.current_file_path, 'result', 'summary')
        self.output_check_point_path = os.path.join(self.current_file_path, 'result', 'check_point')
        self.output_test_features_path = os.path.join(self.current_file_path, 'result', 'test_features')
        self.resnet_checkpoint_file = 'resnet_v2_50.ckpt'
        self.check_path_exist()
        self.checkpoint_name = 'trip_improve_mul.ckpt'


        self.max_step = 30001
        self.num_per_epoch = 10000
        self.num_epochs_per_decay = 30
        self.test_batch_size = 30
        self.change_file_step = 1000

        self.output_feature_dim = 128
        self.dropout = 0.9
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.9
        self.moving_average_decay = 0.999999
        self.return_id_num = 18
        self.image_num_every_id = 4
        self.train_batch_size = self.return_id_num * self.image_num_every_id
        # self.tau1 = 0.6
        # self.tau2 = 0.01
        # self.beta = 0.002
        self.m = 0.4


        with open(self.dataset_train_1000_gallery_csv_file_path, 'rb') as f:
            self.train_1000_gallery_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_train_1000_probe_csv_file_path, 'rb') as f:
            self.train_1000_probe_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_valid_gallery_csv_file_path, 'rb') as f:
            self.valid_gallery_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_valid_probe_csv_file_path, 'rb') as f:
            self.valid_probe_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_predict_gallery_csv_file_path, 'rb') as f:
            self.predict_gallery_num = sum([1 for row in csv.reader(f)])
        with open(self.dataset_predict_probe_csv_file_path, 'rb') as f:
            self.predict_probe_num = sum([1 for row in csv.reader(f)])
        for root, dirs, files in os.walk(self.id_image_path):
            self.id_image_train_num = sum([1 if 'train' in file_name else 0 for file_name in files])
        for root, dirs, files in os.walk(self.id_image_path):
            self.id_image_valid_num = sum([1 if 'valid' in file_name else 0 for file_name in files])


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

    def build(self, resnet_avg_pool, train_test_mode):
        """
        :param train_test_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """
        assert resnet_avg_pool.get_shape().as_list()[1:] == [1, 1, 2048]
        self.resnet_avg_pool_flat = tf.reshape(resnet_avg_pool, [-1, 1 * 1 * 2048])
        self.fc1_add = tf.layers.dense(inputs=self.resnet_avg_pool_flat, units=1024,
                                       kernel_initializer=tf.truncated_normal_initializer(0.0, 0.001), use_bias=False,
                                       name='fc1_add')
        self.bn1_add = tf.layers.batch_normalization(inputs=self.fc1_add, axis=-1, name='bn1_add')
        assert self.bn1_add.get_shape().as_list()[1:] == [1024]
        self.relu1_add = tf.nn.relu(features=self.bn1_add)
        self.fc2_add = tf.layers.dense(inputs=self.relu1_add, units=128,
                                       kernel_initializer=tf.truncated_normal_initializer(0.0, 0.001), use_bias=False,
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
        losses = tf.maximum(m + distmat_intra_max - distmat_inter_min, 0)
        loss = tf.reduce_mean(losses)
        tf.add_to_collection('loss', loss)
        tf.summary.scalar('loss', loss)

        # calculate state
        distmat_intra_mean = tf.reduce_sum(distmat_intra) / tf.reduce_sum(mask_intra)
        distmat_inter_mean = tf.reduce_sum(mask_inter * distmat) / tf.reduce_sum(mask_inter)
        tf.summary.scalar('distmat_intra_mean', distmat_intra_mean)
        tf.summary.scalar('distmat_inter_mean', distmat_inter_mean)
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


    def get_train_image_batch(self, folder_path, file_num, return_id_num, image_num_every_id, change_file = False):
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

    def get_valid_image_batch(self, batch_index, batch_size, folder_path, gallery_flag):
        if gallery_flag:
            try:
                self.valid_gallery_image is None
            except:
                print('valid gallery image first load')
                file_path = os.path.join(folder_path, 'valid_gallery_image_%s_%s_0.pkl' % (IMAGE_HEIGHT, IMAGE_WIDTH))
                with open(file_path, "rb") as f:
                    self.valid_gallery_image = pickle.load(f)
                    for i in range(batch_size):
                        self.valid_gallery_image.append(self.valid_gallery_image[0])

            images = np.array(self.valid_gallery_image[batch_index: batch_index + batch_size], dtype=np.uint8)
            images = images.astype(np.float32) / 255
            # images: [0.0 1.0]
            return images
        else:
            try:
                self.valid_probe_image is None
            except:
                print('valid probe image first load')
                file_path = os.path.join(folder_path, 'valid_probe_image_%s_%s_0.pkl' % (IMAGE_HEIGHT, IMAGE_WIDTH))
                with open(file_path, "rb") as f:
                    self.valid_probe_image = pickle.load(f)
                    for i in range(batch_size):
                        self.valid_probe_image.append(self.valid_probe_image[0])

            images = np.array(self.valid_probe_image[batch_index: batch_index + batch_size], dtype=np.uint8)
            images = images.astype(np.float32) / 255
            # images: [0.0 1.0]
            return images


    def train_batch_inputs(self, dataset_csv_file_path, batch_size, random_flag):

        with tf.name_scope('train_batch_processing'):
            if (os.path.isfile(dataset_csv_file_path) != True):
                raise ValueError('No data files found for this dataset')

            filename_queue = tf.train.string_input_producer([dataset_csv_file_path], shuffle=random_flag)
            reader = tf.TextLineReader()
            _, serialized_example = reader.read(filename_queue)
            ref_image_path, pos_image_path, neg_image_path, order = tf.decode_csv(
                serialized_example, [["ref_image_path"], ["pos_image_path"], ["neg_image_path"], ["order"]])

            # input
            ref_image = tf.read_file(ref_image_path)
            ref = tf.image.decode_jpeg(ref_image, channels=3)
            ref = tf.cast(ref, dtype=tf.float32)

            pos_image = tf.read_file(pos_image_path)
            pos = tf.image.decode_jpeg(pos_image, channels=3)
            pos = tf.cast(pos, dtype=tf.float32)

            neg_image = tf.read_file(neg_image_path)
            neg = tf.image.decode_jpeg(neg_image, channels=3)
            neg = tf.cast(neg, dtype=tf.float32)

            # resized_ref = tf.image.resize_images(ref, (IMAGE_HEIGHT, IMAGE_WIDTH))
            # resized_pos = tf.image.resize_images(pos, (IMAGE_HEIGHT, IMAGE_WIDTH))
            # resized_neg = tf.image.resize_images(neg, (IMAGE_HEIGHT, IMAGE_WIDTH))
            resized_ref = tf.image.resize_images(ref, (IMAGE_HEIGHT, IMAGE_WIDTH + IMAGE_WIDTH / 4))
            resized_pos = tf.image.resize_images(pos, (IMAGE_HEIGHT, IMAGE_WIDTH + IMAGE_WIDTH / 4))
            resized_neg = tf.image.resize_images(neg, (IMAGE_HEIGHT, IMAGE_WIDTH + IMAGE_WIDTH / 4))
            resized_ref = tf.image.crop_to_bounding_box(resized_ref, 0, IMAGE_WIDTH / 8, IMAGE_HEIGHT, IMAGE_WIDTH)
            resized_pos = tf.image.crop_to_bounding_box(resized_pos, 0, IMAGE_WIDTH / 8, IMAGE_HEIGHT, IMAGE_WIDTH)
            resized_neg = tf.image.crop_to_bounding_box(resized_neg, 0, IMAGE_WIDTH / 8, IMAGE_HEIGHT, IMAGE_WIDTH)

            resized_ref = my_preprocess_train(resized_ref, IMAGE_HEIGHT, IMAGE_WIDTH, p_flip=0.2)
            resized_pos = my_preprocess_train(resized_pos, IMAGE_HEIGHT, IMAGE_WIDTH, p_flip=0.2)
            resized_neg = my_preprocess_train(resized_neg, IMAGE_HEIGHT, IMAGE_WIDTH, p_flip=0.2)

            resized_ref = 2.0 * (resized_ref / 255.0) - 1.0
            resized_pos = 2.0 * (resized_pos / 255.0) - 1.0
            resized_neg = 2.0 * (resized_neg / 255.0) - 1.0

            # generate batch
            trains = tf.train.batch(
                [resized_ref, resized_pos, resized_neg, order],
                batch_size=batch_size,
                capacity=1 + 3 * batch_size
            )
            return trains

    def test_batch_inputs(self, dataset_test_csv_file_path, batch_size):

        with tf.name_scope('valid_batch_processing'):
            if (os.path.isfile(dataset_test_csv_file_path) != True):
                raise ValueError('No data files found for this test dataset')

            filename_queue = tf.train.string_input_producer([dataset_test_csv_file_path], shuffle=False)
            reader = tf.TextLineReader()
            _, serialized_example = reader.read(filename_queue)
            test_image_path, test_image_label, order = tf.decode_csv(serialized_example,
                                                                     [["test_image_path"], ["test_image_label"],
                                                                      ["order"]])

            # input
            test_file = tf.read_file(test_image_path)
            test_image = tf.image.decode_jpeg(test_file, channels=3)
            test_image = tf.cast(test_image, dtype=tf.float32)

            # resized_test = tf.image.resize_images(test_image, (IMAGE_HEIGHT, IMAGE_WIDTH))
            resized_test = tf.image.resize_images(test_image, (IMAGE_HEIGHT, IMAGE_WIDTH + IMAGE_WIDTH / 4))
            resized_test = tf.image.crop_to_bounding_box(resized_test, 0, IMAGE_WIDTH / 8, IMAGE_HEIGHT, IMAGE_WIDTH)
            resized_test = 2.0 * (resized_test / 255.0) - 1.0
            # generate batch
            tests = tf.train.batch(
                [resized_test, test_image_label, order],
                batch_size=batch_size,
                capacity=1 + batch_size
            )
            return tests
