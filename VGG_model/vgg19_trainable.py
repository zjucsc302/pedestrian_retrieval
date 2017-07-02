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

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 112


class Train_Flags():
    def __init__(self):

        self.current_file_path = os.path.abspath('.')
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
        self.check_path_exist()
        self.checkpoint_name = 'trip_improve_mine_64.ckpt'

        self.max_step = 30001
        self.num_per_epoch = 10000
        self.num_epochs_per_decay = 30
        self.train_batch_size = 16 # image = 3*train_batch_size
        self.test_batch_size = 30
        self.random_train_input_flag = True

        self.output_feature_dim = 1024
        self.dropout = 0.9
        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.9
        self.moving_average_decay = 0.999999
        self.tau1 = 0.4
        self.tau2 = 0.01
        self.beta = 0.002

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


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.dropout = dropout

    # # def build(self, rgb_scaled, test_rgb_scaled, train_test_mode):
    # def build(self, rgb_scaled, train_test_mode):
    #     """
    #     load variable from npy to build the VGG
    #
    #     :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
    #     :param train_test_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
    #     """
    #
    #     # Convert RGB to BGR
    #     red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    #     assert red.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
    #     assert green.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
    #     assert blue.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
    #     bgr = tf.concat(axis=3, values=[
    #         blue - VGG_MEAN[0],
    #         green - VGG_MEAN[1],
    #         red - VGG_MEAN[2],
    #     ])
    #     assert bgr.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    #
    #     self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
    #     self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
    #     self.pool1 = self.max_pool(self.conv1_2, 'pool1')
    #
    #     self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
    #     self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
    #     self.pool2 = self.max_pool(self.conv2_2, 'pool2')
    #
    #     self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
    #     self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
    #     self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
    #     self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
    #     self.pool3 = self.max_pool(self.conv3_4, 'pool3')
    #
    #     self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
    #     self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
    #     self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
    #     self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
    #     self.pool4 = self.max_pool(self.conv4_4, 'pool4')
    #     '''
    #     self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
    #     self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
    #     self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
    #     self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
    #     self.pool5 = self.max_pool(self.conv5_4, 'pool5')
    #
    #     self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    #     self.relu6 = tf.nn.relu(self.fc6)
    #     if train_mode is not None:
    #         self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
    #     elif self.trainable:
    #         self.relu6 = tf.nn.dropout(self.relu6, self.dropout)
    #
    #     self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
    #     self.relu7 = tf.nn.relu(self.fc7)
    #     if train_mode is not None:
    #         self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
    #     elif self.trainable:
    #         self.relu7 = tf.nn.dropout(self.relu7, self.dropout)
    #
    #     self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")
    #
    #     self.prob = tf.nn.softmax(self.fc8, name="prob")
    #
    #     self.data_dict = None
    #
    #     '''
    #
    #     # self.fc6 = self.fc_layer(self.pool3, 100352, 256, "fc6_new")
    #     self.fc6 = self.fc_layer(self.pool4, 50176, 1024, "fc6_new")
    #     # self.fc6 = self.fc_layer(self.pool5, 14336, 256, "fc6_new")
    #     # self.fc6 = self.fc_layer(self.pool5, 20480, 256, "fc6_new")
    #     self.relu6 = tf.nn.relu(self.fc6)
    #     self.relu6 = tf.cond(train_test_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
    #
    #     self.fc7 = self.fc_layer(self.relu6, 1024, 512, "fc7_new")
    #     # self.relu7 = tf.nn.relu(self.fc7)
    #     # self.relu7 = tf.cond(train_test_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
    #
    #     # l2_normalize
    #     # for the reason of tf.nn.l2_normalize, input has the same dtype with the output, should be float
    #     # self.output = tf.cast(self.fc7, dtype=tf.float32)
    #     # self.output = tf.nn.l2_normalize(self.output, dim=1)
    #     self.output = self.fc7
    #
    #     self.data_dict = None

    # def build(self, rgb_scaled, test_rgb_scaled, train_test_mode):
    def build(self, rgb_scaled, train_test_mode):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_test_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
        assert green.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
        assert blue.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [IMAGE_HEIGHT, IMAGE_WIDTH, 3]

        self.conv1_0 = tf.layers.conv2d(
            inputs=bgr,
            filters=32,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu,
            name='conv1_0')

        self.pool1_0 = tf.layers.max_pooling2d(
            inputs=self.conv1_0,
            pool_size=[2, 2],
            strides=2,
            name='pool1_0')

        self.conv2_0 = tf.layers.conv2d(
            inputs=self.pool1_0,
            filters=32,
            kernel_size=[5, 5],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu,
            name='conv2_0')

        self.pool2_0 = tf.layers.max_pooling2d(
            inputs=self.conv2_0,
            pool_size=[2, 2],
            strides=2,
            name='pool2')

        self.pool2_flat = tf.reshape(self.pool2_0, [-1, 28 * 14 * 32])
        # self.pool2_flat = tf.cond(train_test_mode, lambda: tf.nn.dropout(self.pool2_flat, self.dropout), lambda: self.pool2_flat)

        self.fc3_0 = tf.layers.dense(inputs=self.pool2_flat,
                                     units=1024,
                                     activation=tf.nn.relu,
                                     name='fc3_0')
        self.fc3_0 = tf.cond(train_test_mode, lambda: tf.nn.dropout(self.fc3_0, self.dropout), lambda: self.fc3_0)

        self.output = self.fc3_0

        self.data_dict = None

    # improved max triplet loss
    def calc_loss(self, logits, tau1, tau2, beta):
        # # for the reason of tf.nn.l2_normalize, input has the same dtype with the output, should be float
        logits = tf.cast(logits, dtype=tf.float32)
        logits = tf.nn.l2_normalize(logits, dim=1)

        split_refs, split_poss, split_negs = tf.split(logits, num_or_size_splits=3, axis=0)

        dist_ref_to_pos = tf.norm(split_refs - split_poss, 2, 1)
        dist_ref_to_neg = tf.norm(split_refs - split_negs, 2, 1)
        dist_pos_to_neg = tf.norm(split_poss - split_negs, 2, 1)

        max_dis_in_triplet = tf.maximum(dist_ref_to_pos - dist_ref_to_neg, dist_ref_to_pos - dist_pos_to_neg)
        inter_const = tf.maximum(max_dis_in_triplet + tau1, 0.0)
        intra_const = tf.maximum(dist_ref_to_pos - tau2, 0.0)
        costs = inter_const + beta * intra_const
        tf.add_to_collection('losses', costs)

        tf.summary.scalar('inter_const_mean', tf.reduce_mean(dist_ref_to_neg))
        tf.summary.scalar('intra_const_mean', tf.reduce_mean(dist_ref_to_pos))
        tf.summary.scalar('inter_const_min', tf.reduce_min(dist_ref_to_neg))
        tf.summary.scalar('intra_const_max', tf.reduce_max(dist_ref_to_pos))
        accuracy = tf.reduce_mean(tf.cast(dist_ref_to_pos < dist_ref_to_neg, dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)

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

            resized_ref = my_preprocess_train(resized_ref, IMAGE_HEIGHT, IMAGE_WIDTH)
            resized_pos = my_preprocess_train(resized_pos, IMAGE_HEIGHT, IMAGE_WIDTH)
            resized_neg = my_preprocess_train(resized_neg, IMAGE_HEIGHT, IMAGE_WIDTH)


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

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


        # def fc_layer_fine_tune(self, bottom, in_size, out_size, name):
        #     with tf.variable_scope(name):
        #         weights, biases = self.get_fc_var_fine_tune(in_size, out_size, name)
        #
        #         x = tf.reshape(bottom, [-1, in_size])
        #         fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        #
        #         return fc
        #
        # def get_fc_var_fine_tune(self, in_size, out_size, name):
        #     weights = self._variable_with_weight_decay('weights', shape=[in_size, out_size],
        #         stddev=0.001, wd=0.0, trainable=True)
        #     biases = self._variable_on_gpu('biases', [out_size], tf.constant_initializer(0.001))
        #
        #     return weights, biases
        #
        # def _variable_on_gpu(self, name, shape, initializer):
        #     var = tf.get_variable(name, shape, initializer=initializer)
        #     return var
        #
        # def _variable_with_weight_decay(self, name, shape, stddev, wd, trainable=True):
        #     var = self._variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        #     return var
        #
        # def conv_layer_fine_tune(self, bottom, in_channels, out_channels, name):
        #     with tf.variable_scope(name):
        #         filt, conv_biases = self.get_conv_var_fine_tune(3, in_channels, out_channels, name)
        #
        #         conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        #         bias = tf.nn.bias_add(conv, conv_biases)
        #         relu = tf.nn.relu(bias)
        #
        #         return relu
        #
        # def get_conv_var_fine_tune(self, filter_size, in_channels, out_channels, name):
        #
        #     filters = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        #     biases = tf.truncated_normal([out_channels], .0, .001)
        #
        #     return filters, biases
