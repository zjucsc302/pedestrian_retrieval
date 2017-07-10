import os.path
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from res_trainable import ResnetReid
from res_trainable import Train_Flags
from res_trainable import IMAGE_HEIGHT, IMAGE_WIDTH
slim = tf.contrib.slim
from resnet_v2 import resnet_v2_50, resnet_arg_scope



train_flags = Train_Flags()


def _model_loss(vgg_class):
    with tf.variable_scope(tf.get_variable_scope()):
        vgg_class.calc_loss(vgg_class.output, train_flags.return_id_num, train_flags.image_num_every_id , train_flags.m)
    loss = tf.add_n(tf.get_collection('loss'))
    # loss_mean = evg_loss(loss_mean, 0.9)
    return loss


def evg_loss(loss, decay):
    loss_averages = tf.train.ExponentialMovingAverage(decay, name='avg')
    loss_averages_op = loss_averages.apply([loss])
    with tf.control_dependencies([loss_averages_op]):
        loss = tf.identity(loss)
    return loss


def train(retain_flag=True, start_step=0):
    print('train(%s)' % (retain_flag))

    with tf.Graph().as_default():
        # define placeholder
        train_mode = tf.placeholder(tf.bool, name='train_mode')
        input_batch = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], 'input_batch')

        # define model
        resnet_reid = ResnetReid()
        with slim.arg_scope(resnet_arg_scope()):
            # input_batch: [batch, height, width, 3] values scaled [0.0, 1.0], dtype = tf.float32
            resnet_avg_pool, end_points = resnet_v2_50(input_batch, is_training=False, global_pool=True)
        # Define the scopes that you want to exclude for restoration
        variables_to_restore = slim.get_variables_to_restore(exclude=['beta2_power'])
        resnet_reid.build(resnet_avg_pool, train_mode)

        # define loss
        loss = tf.cond(train_mode, lambda: _model_loss(resnet_reid), lambda: tf.constant([0.0], dtype=tf.float32), 'chose_loss')

        # Create an optimizer that performs gradient descent.
        global_step = get_or_create_global_step()
        num_batches_per_epoch = (train_flags.num_per_epoch / train_flags.train_batch_size)
        decay_steps = int(num_batches_per_epoch * train_flags.num_epochs_per_decay)
        lr = tf.train.exponential_decay(train_flags.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        train_flags.learning_rate_decay_factor,
                                        staircase=True)
        vars_to_optimize = [v for v in tf.trainable_variables() if ('add' in v.name)]
        print '\nvariables to optimize'
        for v in vars_to_optimize:
            print v.name, v.get_shape().as_list()
            tf.summary.histogram(v.name, v)
        opt = tf.train.AdamOptimizer(lr)

        grads = opt.compute_gradients(loss, var_list=vars_to_optimize)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op)

        # define sess
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # define summary and saver
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(train_flags.output_summary_path, graph=sess.graph)
        saver = tf.train.Saver()

        # retrain or continue
        if retain_flag:
            saver_resnet = tf.train.Saver(variables_to_restore)
            saver_resnet.restore(sess, train_flags.resnet_checkpoint_file)
        else:
            # load checkpoint
            ckpt = tf.train.get_checkpoint_state(train_flags.output_check_point_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('load checkpoint')

        print('start training')
        for step in range(start_step, train_flags.max_step):
            # input image
            if step % train_flags.change_file_step == 0:
                change_file_flag = True
            else:
                change_file_flag = False
            bath = resnet_reid.get_train_image_batch(train_flags.id_image_path, train_flags.id_image_train_num,
                                                     train_flags.return_id_num, train_flags.image_num_every_id,
                                                     change_file=change_file_flag)

            # start run
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss], feed_dict={input_batch: bath, train_mode: True})
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = train_flags.train_batch_size / float(duration)
                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

            if step % 100 == 0:
                summary_str, feature = sess.run([summary_op, resnet_reid.output],feed_dict={input_batch: bath, train_mode: True})
                print('feature abs mean: %s' % (np.mean(np.abs(feature))))
                summary_str = sess.run(summary_op, feed_dict={input_batch: bath, train_mode: True})
                summary_writer.add_summary(summary_str, step)

            if step % 5000 == 0 or (step + 1) == train_flags.max_step:
                # Save the model checkpoint periodically.
                checkpoint_path = os.path.join(train_flags.output_check_point_path, train_flags.checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=step)

                # do test: send every image in valid_gallery.csv and valid_probe.csv, then get feature vector
                # save all feature vector in npy
                def valid(gallery_flag):
                    # get feature batch
                    features_num = train_flags.valid_gallery_num if gallery_flag else train_flags.valid_probe_num
                    features = np.zeros((features_num, train_flags.output_feature_dim), dtype=float)
                    batch_len = train_flags.test_batch_size
                    batch_name = 'valid_gallery_batch_index: ' if gallery_flag else 'valid_probe_batch_index: '
                    end_len = features_num % batch_len
                    for batch_index in range(0, features_num - end_len, batch_len):
                        bath = resnet_reid.get_test_image_batch(batch_index, batch_len, train_flags.id_image_path, 'valid', gallery_flag)
                        batch_feature = sess.run(resnet_reid.output, feed_dict={input_batch: bath, train_mode: False})
                        features[batch_index: batch_index + batch_len, :] = batch_feature
                        print(batch_name + str(batch_index) + '-' + str(batch_index + batch_len - 1))
                    if end_len != 0:
                        batch_index += batch_len
                        bath = resnet_reid.get_test_image_batch(batch_index, batch_len, train_flags.id_image_path, 'valid', gallery_flag)
                        batch_feature = sess.run(resnet_reid.output, feed_dict={input_batch: bath, train_mode: False})
                        features[batch_index: batch_index + end_len, :] = batch_feature[:end_len]
                        print(batch_name + str(batch_index) + '-' + str(batch_index + end_len - 1))

                    # save feature
                    features_npy_name = 'valid_gallery_features_step-%d.npy' if gallery_flag else 'valid_probe_features_step-%d.npy'
                    features_npy_path = os.path.join(train_flags.output_test_features_path, features_npy_name % step)
                    np.save(features_npy_path, features)

                valid(False)
                valid(True)
        sess.close()


def generate_features(predict_flag, gallery_flag):
    print('generate_features(%s, %s)' % (predict_flag, gallery_flag))

    with tf.Graph().as_default():
        # define placeholder
        train_mode = tf.placeholder(tf.bool, name='train_mode')
        input_batch = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], 'input_batch')

        # define model
        resnet_reid = ResnetReid()
        with slim.arg_scope(resnet_arg_scope()):
            # input_batch: [batch, height, width, 3] values scaled [0.0, 1.0], dtype = tf.float32
            resnet_avg_pool, end_points = resnet_v2_50(input_batch, is_training=False, global_pool=True)
        resnet_reid.build(resnet_avg_pool, train_mode)

        # define sess
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # load checkpoint
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(train_flags.output_check_point_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('load checkpoint')

            # define parameter
            if gallery_flag and predict_flag:
                features_num = train_flags.predict_gallery_num
                features_npy_name = 'predict_gallery_features.npy'
                get_image_batch = resnet_reid.get_predict_image_batch
                file_title = 'predict'
            elif gallery_flag and (not predict_flag):
                features_num = train_flags.train_200_gallery_num
                features_npy_name = 'train_200_gallery_features.npy'
                get_image_batch = resnet_reid.get_test_image_batch
                file_title = 'train_200'
            elif (not gallery_flag) and predict_flag:
                features_num = train_flags.predict_probe_num
                features_npy_name = 'predict_probe_features.npy'
                get_image_batch = resnet_reid.get_predict_image_batch
                file_title = 'predict'
            else:
                features_num = train_flags.train_200_probe_num
                features_npy_name = 'train_200_probe_features.npy'
                get_image_batch = resnet_reid.get_test_image_batch
                file_title = 'train_200'

            print('start generate features')
            # get feature batch
            features = np.zeros((features_num, train_flags.output_feature_dim), dtype=float)
            batch_len = train_flags.test_batch_size
            end_len = features_num % batch_len
            for batch_index in range(0, features_num - end_len, batch_len):
                bath = get_image_batch(batch_index, batch_len, train_flags.id_image_path, file_title, gallery_flag)
                batch_feature = sess.run(resnet_reid.output, feed_dict={input_batch: bath, train_mode: False})
                features[batch_index: batch_index + batch_len, :] = batch_feature
                print('batch_index: ' + str(batch_index) + '-' + str(batch_index + batch_len - 1))
            if end_len != 0:
                batch_index += batch_len
                bath = get_image_batch(batch_index, batch_len, train_flags.id_image_path, file_title, gallery_flag)
                batch_feature = sess.run(resnet_reid.output, feed_dict={input_batch: bath, train_mode: False})
                features[batch_index: batch_index + end_len, :] = batch_feature[:end_len]
                print('batch_index: ' + str(batch_index) + '-' + str(batch_index + end_len - 1))

            # save feature
            features_npy_path = os.path.join(train_flags.output_test_features_path, features_npy_name)
            np.save(features_npy_path, features)


if __name__ == '__main__':
    train(retain_flag=True, start_step=1)
    # generate_features(predict_flag=True, gallery_flag=True)
    # generate_features(predict_flag=True, gallery_flag=False)
    # generate_features(predict_flag=False, gallery_flag=True)
    # generate_features(predict_flag=False, gallery_flag=False)

    # res = ResnetReid()
    # print res.get_train_image_batch(train_flags.id_image_path, train_flags.id_image_train_num, train_flags.return_id_num, train_flags.image_num_every_id)
    # print res.get_train_image_batch(train_flags.id_image_path, train_flags.id_image_train_num, train_flags.return_id_num, train_flags.image_num_every_id)
    # print res.get_valid_image_batch(0, 72, train_flags.id_image_path, gallery_flag=False)

