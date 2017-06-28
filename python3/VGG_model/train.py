from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

from vgg19_trainable import Vgg19
from vgg19_trainable import Train_Flags

train_flags = Train_Flags()


def _model_loss(vgg_class):
    # Compute the moving average of all losses
    with tf.variable_scope(tf.get_variable_scope()):
        vgg_class.calc_loss(vgg_class.output, train_flags.tau1, train_flags.tau2, train_flags.beta)
    losses = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # Compute the moving average of total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply([losses])
    with tf.control_dependencies([loss_averages_op]):
        losses = tf.identity(losses)
    return losses


def train(retain_flag=True, start_step=0):
    print('train(%s)' % (retain_flag))
    # train model, generate valid features
    with tf.Graph().as_default():

        # build a VGG19 class object
        train_mode = tf.placeholder(tf.bool)
        gallery_mode = tf.placeholder(tf.bool)  # gallery or probe

        # define model
        vgg = Vgg19(vgg19_npy_path='./vgg19.npy', train_test_mode=train_mode)

        # define input data
        refs_batch, poss_batch, negs_batch, train_orders_batch = vgg.train_batch_inputs(
            train_flags.dataset_train_csv_file_path,
            train_flags.train_batch_size)
        train_batch = tf.concat([refs_batch, poss_batch, negs_batch], 0)
        valid_gallery_batch, valid_gallery_label, valid_gallery_order = vgg.test_batch_inputs(
            train_flags.dataset_valid_gallery_csv_file_path, train_flags.test_batch_size)
        valid_probe_batch, valid_probe_label, valid_probe_order = vgg.test_batch_inputs(
            train_flags.dataset_valid_probe_csv_file_path,
            train_flags.test_batch_size)
        test_batch = tf.cond(gallery_mode, lambda: valid_gallery_batch, lambda: valid_probe_batch)
        input_batch = tf.cond(vgg.train_test_mode, lambda: train_batch, lambda: test_batch)

        # build model
        with tf.variable_scope(tf.get_variable_scope()):
            vgg.build(input_batch, vgg.train_test_mode)
            # loss
            losses = tf.cond(train_mode, lambda: _model_loss(vgg), lambda: tf.constant([0], dtype=tf.float32))
            loss_mean = tf.reduce_mean(losses)
            loss_max = tf.reduce_max(losses)
            tf.summary.scalar('loss_mean', loss_mean)
            tf.summary.scalar('loss_max', loss_max)

        # Create an optimizer that performs gradient descent.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = (train_flags.num_per_epoch / train_flags.train_batch_size)
        decay_steps = int(num_batches_per_epoch * train_flags.num_epochs_per_decay)
        lr = tf.train.exponential_decay(train_flags.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        train_flags.learning_rate_decay_factor,
                                        staircase=True)
        vars_to_optimize = [v for v in tf.trainable_variables() if
                            (v.name.startswith('fc') | v.name.startswith('conv__'))]
        print ('\nvariables to optimize')
        for v in vars_to_optimize:
            print (v.name, v.get_shape().as_list())
            tf.summary.histogram(v.name, v)
        opt = tf.train.AdamOptimizer(lr)

        # grads = opt.compute_gradients(loss, var_list=vars_to_optimize)
        grads = opt.compute_gradients(loss_mean, var_list=vars_to_optimize)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op)

        # define sess
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_flags.output_summary_path, graph=sess.graph)
        saver = tf.train.Saver(tf.global_variables())

        # retrain or continue
        if not retain_flag:
            # load checkpoint
            ckpt = tf.train.get_checkpoint_state(train_flags.output_check_point_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('load checkpoint')

        print('start training')
        for step in range(start_step, train_flags.max_step):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss_mean], feed_dict={train_mode: True, gallery_mode: True})
            # print pool5.shape
            # _, loss_value, feature = sess.run([train_op, loss_mean, vgg.output], feed_dict={train_mode: True, gallery_mode: True})
            # print('feature abs mean: %s' % (np.mean(np.abs(feature))))
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = train_flags.train_batch_size / float(duration)
                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

            if step % 100 == 0:
                summary_str, feature = sess.run([summary_op, vgg.output],
                                                feed_dict={train_mode: True, gallery_mode: True})
                print('feature abs mean: %s' % (np.mean(np.abs(feature))))
                summary_str = sess.run(summary_op, feed_dict={train_mode: True, gallery_mode: True})
                summary_writer.add_summary(summary_str, step)

            if step % 5000 == 0 or (step + 1) == train_flags.max_step:
                # if (step % 5000 == 0 or (step + 1) == train_flags.max_step) and step != 0:
                # Save the model checkpoint periodically.
                checkpoint_path = os.path.join(train_flags.output_check_point_path, train_flags.checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=step)

                # do test: send every image in valid_gallery.csv and valid_probe.csv, then get feature vector
                # save all feature vector in npy
                def valid(gallery_flag):
                    # get feature batch
                    features_num = train_flags.valid_gallery_num if gallery_flag else train_flags.valid_probe_num
                    features = np.zeros((features_num, train_flags.output_feature_dim), dtype=float)
                    labels = np.zeros(features_num, dtype=float)
                    batch_len = train_flags.test_batch_size
                    batch_name = 'valid_gallery_batch_index: ' if gallery_flag else 'valid_probe_batch_index: '
                    valid_label = valid_gallery_label if gallery_flag else valid_probe_label
                    end_len = features_num % batch_len
                    for batch_index in range(0, features_num - end_len, batch_len):
                        batch_feature, batch_label = sess.run([vgg.output, valid_label],
                                                              feed_dict={train_mode: False,
                                                                         gallery_mode: gallery_flag})
                        features[batch_index: batch_index + batch_len, :] = batch_feature
                        labels[batch_index: batch_index + batch_len] = batch_label
                        print(batch_name + str(batch_index) + '-' + str(batch_index + batch_len - 1))
                    if end_len != 0:
                        batch_feature, batch_label = sess.run([vgg.output, valid_label],
                                                              feed_dict={train_mode: False,
                                                                         gallery_mode: gallery_flag})
                        features[batch_index + batch_len: batch_index + batch_len + end_len, :] = batch_feature[
                                                                                                  :end_len]
                        labels[batch_index + batch_len: batch_index + batch_len + end_len] = batch_label[:end_len]
                        print(batch_name + str(batch_index + batch_len) + '-' + str(
                            batch_index + batch_len + end_len - 1))

                    # save feature
                    features_csv_name = 'valid_gallery_features_step-%d.npy' if gallery_flag else 'valid_probe_features_step-%d.npy'
                    features_csv_path = os.path.join(train_flags.output_test_features_path, features_csv_name % step)
                    np.save(features_csv_path, features)
                    labels_csv_name = 'valid_gallery_labels_step-%d.npy' if gallery_flag else 'valid_probe_labels_step-%d.npy'
                    labels_csv_path = os.path.join(train_flags.output_test_features_path, labels_csv_name % step)
                    np.save(labels_csv_path, labels)

                valid(False)
                valid(True)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def generate_features(predict_flag, gallery_flag):
    print('generate_features(%s, %s)' % (predict_flag, gallery_flag))
    # generate predict features
    with tf.Graph().as_default():
        # build a VGG19 class object
        vgg = Vgg19(train_test_mode=tf.constant(False, tf.bool))

        # define parameter
        if gallery_flag and predict_flag:
            input_path = train_flags.dataset_predict_gallery_csv_file_path
            features_num = train_flags.predict_gallery_num
            features_csv_name = 'predict_gallery_features.npy'
            labels_csv_name = 'predict_gallery_labels.npy'
            orders_csv_name = 'predict_gallery_orders.npy'
        elif gallery_flag and (not predict_flag):
            input_path = train_flags.dataset_train_1000_gallery_csv_file_path
            features_num = train_flags.train_1000_gallery_num
            features_csv_name = 'train_1000_gallery_features.npy'
            labels_csv_name = 'train_1000_gallery_labels.npy'
            orders_csv_name = 'train_1000_gallery_orders.npy'
        elif (not gallery_flag) and predict_flag:
            input_path = train_flags.dataset_predict_probe_csv_file_path
            features_num = train_flags.predict_probe_num
            features_csv_name = 'predict_probe_features.npy'
            labels_csv_name = 'predict_probe_labels.npy'
            orders_csv_name = 'predict_probe_orders.npy'
        else:
            input_path = train_flags.dataset_train_1000_probe_csv_file_path
            features_num = train_flags.train_1000_probe_num
            features_csv_name = 'train_1000_probe_features.npy'
            labels_csv_name = 'train_1000_probe_labels.npy'
            orders_csv_name = 'train_1000_probe_orders.npy'

        # build model
        predict_batch, predict_label, predict_order = vgg.test_batch_inputs(input_path, train_flags.test_batch_size)
        with tf.variable_scope(tf.get_variable_scope()):
            vgg.build(predict_batch, vgg.train_test_mode)

        # run predict
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # load checkpoint
            ckpt = tf.train.get_checkpoint_state(train_flags.output_check_point_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # start the queue runners
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print('start generate features')
            # get feature batch
            features = np.zeros((features_num, train_flags.output_feature_dim), dtype=float)
            labels = np.zeros(features_num, dtype=float)
            orders = np.zeros(features_num, dtype=float)
            batch_len = train_flags.test_batch_size
            end_len = features_num % batch_len
            for batch_index in range(0, features_num - end_len, batch_len):
                batch_feature, batch_label, batch_order = sess.run([vgg.output, predict_label, predict_order])
                features[batch_index: batch_index + batch_len, :] = batch_feature
                labels[batch_index: batch_index + batch_len] = batch_label
                orders[batch_index: batch_index + batch_len] = batch_order
                print('batch_index: ' + str(batch_index) + '-' + str(batch_index + batch_len - 1))
            if end_len != 0:
                batch_feature, batch_label, batch_order = sess.run([vgg.output, predict_label, predict_order])
                features[batch_index + batch_len: batch_index + batch_len + end_len, :] = batch_feature[:end_len]
                labels[batch_index + batch_len: batch_index + batch_len + end_len] = batch_label[:end_len]
                orders[batch_index + batch_len: batch_index + batch_len + end_len] = batch_order[:end_len]
                print('batch_index: ' + str(batch_index + batch_len) + '-' + str(batch_index + batch_len + end_len - 1))

            # save feature
            features_csv_path = os.path.join(train_flags.output_test_features_path, features_csv_name)
            np.save(features_csv_path, features)
            labels_csv_path = os.path.join(train_flags.output_test_features_path, labels_csv_name)
            np.save(labels_csv_path, labels)
            orders_csv_path = os.path.join(train_flags.output_test_features_path, orders_csv_name)
            np.save(orders_csv_path, orders)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train(retain_flag=True, start_step=1)
    # generate_features(predict_flag=True, gallery_flag=True)
    # generate_features(predict_flag=True, gallery_flag=False)
    # generate_features(predict_flag=False, gallery_flag=True)
    # generate_features(predict_flag=False, gallery_flag=False)
