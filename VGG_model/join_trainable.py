import tensorflow as tf

import numpy as np
from functools import reduce
import os
from tensorflow.python.platform import gfile
import csv
from vgg_preprocessing import my_preprocess_train

# VGG_MEAN = [103.939, 116.779, 123.68]
# VGG_MEAN = [98.200, 98.805, 102.044]
VGG_MEAN = [98.3, 98.5, 101.1]

# IMAGE_HEIGHT = 224
# IMAGE_WIDTH = 112

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 60


class Train_Flags():
    def __init__(self):

        self.current_file_path = os.path.abspath('.')
        self.dataset_train_csv_file_path = os.path.abspath('../data/train_triplet_pair.csv')
        self.dataset_train_1000_gallery_csv_file_path = os.path.abspath('../data/train_1000_gallery.csv')
        self.dataset_train_1000_probe_csv_file_path = os.path.abspath('../data/train_1000_probe.csv')
        self.dataset_valid_gallery_csv_file_path = os.path.abspath('../data/valid_gallery.csv')
        self.dataset_valid_probe_csv_file_path = os.path.abspath('../data/valid_probe.csv')
        self.dataset_valid_repeat_probe_csv_file_path = os.path.abspath('../data/valid_repeat_probe.csv')
        self.dataset_predict_gallery_csv_file_path = os.path.abspath('../data/predict_gallery.csv')
        self.dataset_predict_repeat_gallery_csv_file_path = os.path.abspath('../data/predict_repeat_gallery.csv')
        self.dataset_predict_probe_csv_file_path = os.path.abspath('../data/predict_probe.csv')
        self.dataset_predict_repeat_probe_csv_file_path = os.path.abspath('../data/predict_repeat_probe.csv')
        self.output_summary_path = os.path.join(self.current_file_path, 'result', 'summary')
        self.output_check_point_path = os.path.join(self.current_file_path, 'result', 'check_point')
        self.output_test_features_path = os.path.join(self.current_file_path, 'result', 'test_features')
        self.output_test_distance_path = os.path.join(self.current_file_path, 'result', 'distance')
        self.check_path_exist()
        self.checkpoint_name = 'trip_improve_joint.ckpt'

        self.max_step = 30001
        self.num_per_epoch = 10000
        self.num_epochs_per_decay = 30
        self.train_batch_size = 64  # image = 3*train_batch_size
        self.test_batch_size = 120
        self.random_train_input_flag = True

        # self.output_feature_dim = 800
        self.dropout = 0.9
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.9
        self.moving_average_decay = 0.999999
        self.tau1 = 0.6
        self.tau2 = 0.01
        self.beta = 0.002
        self.alpha = 0.0005
        self.epsilon = 1.0
        self.eta = 1.0
        self.lambda_ = 5.0

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

    def check_path_exist(self):
        if not gfile.Exists(self.output_summary_path):
            gfile.MakeDirs(self.output_summary_path)
        if not gfile.Exists(self.output_check_point_path):
            gfile.MakeDirs(self.output_check_point_path)
        if not gfile.Exists(self.output_test_features_path):
            gfile.MakeDirs(self.output_test_features_path)


class Join:
    def __init__(self, vgg19_npy_path=None, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.dropout = dropout
        self.omega = tf.Variable([0.0001 for i in range(1000)], dtype=tf.float32)

    def build(self, rgb_scaled, train_test_mode):
        """
        :param rgb: rgb image [batch, height, width, 3] values scaled [0.0, 255.0]
        :param train_test_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [160, 60, 1]
        assert green.get_shape().as_list()[1:] == [160, 60, 1]
        assert blue.get_shape().as_list()[1:] == [160, 60, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [160, 60, 3]

        # train: r p n, test: p g
        self.sh_conv1 = tf.layers.conv2d(bgr, 32, [5, 5], 1, 'valid', activation=tf.nn.relu, name='sh_conv1')
        assert self.sh_conv1.get_shape().as_list()[1:] == [156, 56, 32]
        self.sh_pool1 = tf.layers.max_pooling2d(self.sh_conv1, [3, 3], 3, padding='same', name='sh_pool1')
        assert self.sh_pool1.get_shape().as_list()[1:] == [52, 19, 32]
        self.sh_conv2 = tf.layers.conv2d(self.sh_pool1, 64, [3, 3], 1, 'valid', activation=tf.nn.relu, name='sh_conv2')
        assert self.sh_conv2.get_shape().as_list()[1:] == [50, 17, 64]
        self.sh_pool2 = tf.layers.max_pooling2d(self.sh_conv2, [2, 2], 2, padding='same', name='sh_pool2')
        assert self.sh_pool2.get_shape().as_list()[1:] == [25, 9, 64]

        # train: r p n, test: p g
        self.s_conv = tf.layers.conv2d(self.sh_pool2, 32, [3, 3], 1, 'valid', activation=tf.nn.relu, name='s_conv')
        assert self.s_conv.get_shape().as_list()[1:] == [23, 7, 32]
        self.s_pool = tf.layers.max_pooling2d(self.s_conv, [2, 2], 2, padding='same', name='s_pool')
        assert self.s_pool.get_shape().as_list()[1:] == [12, 4, 32]
        self.s_pool_flat = tf.reshape(self.s_pool, [-1, 12 * 4 * 32], 's_pool_flat')
        self.s_fc1 = tf.layers.dense(inputs=self.s_pool_flat, units=1000, name='s_fc1')
        self.s_fc2 = tf.layers.dense(inputs=self.s_fc1, units=500, name='s_fc2')

        # train: rp rn, test: pg
        self.concat = tf.cond(train_test_mode, lambda: self.train_concat(), lambda: self.test_concat())
        self.c_conv = tf.layers.conv2d(self.concat, 32, [3, 3], 1, 'valid', activation=tf.nn.relu, name='c_conv')
        assert self.c_conv.get_shape().as_list()[1:] == [23, 7, 32]
        self.c_pool = tf.layers.max_pooling2d(self.c_conv, [2, 2], 2, padding='same', name='c_pool')
        assert self.c_pool.get_shape().as_list()[1:] == [12, 4, 32]
        self.c_pool_flat = tf.reshape(self.c_pool, [-1, 12 * 4 * 32], 'c_pool_flat')
        self.c_fc = tf.layers.dense(inputs=self.c_pool_flat, units=1000, name='c_fc')

        self.s_feature = self.s_fc2
        self.c_feature = self.c_fc
        self.data_dict = None

    def train_concat(self):
        # train: rp rn
        self.refs, self.poss, self.negs = tf.split(self.sh_pool2, num_or_size_splits=3, axis=0)
        self.concat_rp = tf.concat([self.refs, self.poss], 3, 'concat_rp')
        assert self.concat_rp.get_shape().as_list()[1:] == [25, 9, 128]
        self.concat_rn = tf.concat([self.refs, self.negs], 3, 'concat_rn')
        assert self.concat_rn.get_shape().as_list()[1:] == [25, 9, 128]
        self.concat_rprn = tf.concat([self.concat_rp, self.concat_rn], 0, 'concat_rprn')
        assert self.concat_rprn.get_shape().as_list()[1:] == [25, 9, 128]
        return self.concat_rprn

    def test_concat(self):
        # test: pg
        self.probe, self.gallery = tf.split(self.sh_pool2, num_or_size_splits=2, axis=0)
        self.concat_pg = tf.concat([self.probe, self.gallery], 3, 'concat_rp')
        assert self.concat_pg.get_shape().as_list()[1:] == [25, 9, 128]
        return self.concat_pg

    # improved max triplet loss
    def calc_loss(self, s_feature, c_feature, tau1, tau2, beta, alpha, epsilon, eta, lambda_):
        # loss SIR
        s_feature = tf.cast(s_feature, dtype=tf.float32)
        s_feature = tf.nn.l2_normalize(s_feature, dim=1)
        split_refs, split_poss, split_negs = tf.split(s_feature, num_or_size_splits=3, axis=0)
        dist_ref_to_pos = tf.norm(split_refs - split_poss, 2, 1, keep_dims=True)
        dist_ref_to_neg = tf.norm(split_refs - split_negs, 2, 1, keep_dims=True)
        dist_pos_to_neg = tf.norm(split_poss - split_negs, 2, 1, keep_dims=True)

        max_dis_in_triplet = tf.maximum(dist_ref_to_pos ** 2 - dist_ref_to_neg ** 2,
                                        dist_ref_to_pos ** 2 - dist_pos_to_neg ** 2)
        inter_const = tf.maximum(max_dis_in_triplet + tau1, 0.0)
        intra_const = tf.maximum(dist_ref_to_pos ** 2 - tau2, 0.0)
        loss_SIRs = inter_const + beta * intra_const
        loss_SIR_mean = tf.reduce_mean(loss_SIRs)
        loss_SIR_max = tf.reduce_max(loss_SIRs)
        tf.summary.scalar('SIR_loss_mean', loss_SIR_mean)
        tf.summary.scalar('SIR_loss_max', loss_SIR_max)
        tf.summary.scalar('SIR_inter_mean', tf.reduce_mean(dist_ref_to_neg))
        tf.summary.scalar('SIR_intra_mean', tf.reduce_mean(dist_ref_to_pos))
        tf.summary.scalar('SIR_inter_min', tf.reduce_min(dist_ref_to_neg))
        tf.summary.scalar('SIR_intra_max', tf.reduce_max(dist_ref_to_pos))
        # loss CIR
        c_feature = tf.cast(c_feature, dtype=tf.float32)
        split_rps, split_rns = tf.split(c_feature, num_or_size_splits=2, axis=0)
        part1 = alpha / 2.0  * tf.norm(self.omega, 2, 0) ** 2
        dis_rps = tf.reduce_sum(self.omega * split_rps, axis=1, keep_dims=True)
        dis_rns = tf.reduce_sum(self.omega * split_rns, axis=1, keep_dims=True)
        part2 = tf.maximum(dis_rps - dis_rns + epsilon, 0.0)
        loss_CIRs = part1 + part2
        loss_CIR_mean = tf.reduce_mean(loss_CIRs)
        loss_CIR_max = tf.reduce_max(loss_CIRs)
        tf.summary.scalar('CIR_loss_mean', loss_CIR_mean)
        tf.summary.scalar('CIR_loss_max', loss_CIR_max)
        tf.summary.scalar('CIR_inter_mean', tf.reduce_mean(dis_rns))
        tf.summary.scalar('CIR_intra_mean', tf.reduce_mean(dis_rps))
        tf.summary.scalar('CIR_inter_min', tf.reduce_min(dis_rns))
        tf.summary.scalar('CIR_intra_max', tf.reduce_max(dis_rps))
        # loss
        loss = loss_SIR_mean + eta * loss_CIR_mean
        tf.add_to_collection('loss', loss)
        # temp accuracy
        compare = self.predict_dist(dist_ref_to_pos, split_rps, lambda_) < self.predict_dist(dist_ref_to_neg, split_rns, lambda_)
        accuracy = tf.reduce_mean(tf.cast(compare, dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    def predict_dist(self, fij, gij, lambda_):
        return fij ** 2 + lambda_ * tf.reduce_sum(self.omega * gij, axis=1, keep_dims=True)

    def get_predict_dist(self, lambda_):
        s_feature = tf.cast(self.s_feature, dtype=tf.float32)
        s_feature = tf.nn.l2_normalize(s_feature, dim=1)
        s_probes, s_gallerys = tf.split(s_feature, num_or_size_splits=2, axis=0)
        s_dist_p_g = tf.norm(s_probes - s_gallerys, 2, 1, keep_dims=True)
        return self.predict_dist(s_dist_p_g, self.c_feature, lambda_)

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

            # generate batch
            tests = tf.train.batch(
                [resized_test, test_image_label, order],
                batch_size=batch_size,
                capacity=1 + batch_size
            )
            return tests