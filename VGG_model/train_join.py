from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

from join_trainable import Join
from join_trainable import Train_Flags
from cmc import mAP
import csv

train_flags = Train_Flags()


def _model_loss(vgg_class):
    # Compute the moving average of all losses
    with tf.variable_scope(tf.get_variable_scope()):
        vgg_class.calc_loss(vgg_class.s_feature, vgg_class.c_feature, train_flags.tau1, train_flags.tau2,
                            train_flags.beta,
                            train_flags.alpha, train_flags.epsilon, train_flags.eta, train_flags.lambda_)
        losses = tf.add_n(tf.get_collection('loss'), name='total_loss')
    # loss_mean = evg_loss(loss_mean, 0.9)
    return losses


def evg_loss(loss, decay):
    loss_averages = tf.train.ExponentialMovingAverage(decay, name='avg')
    loss_averages_op = loss_averages.apply([loss])
    with tf.control_dependencies([loss_averages_op]):
        loss = tf.identity(loss)
    return loss


def train(retain_flag=True, start_step=0):
    print('train(%s)' % (retain_flag))
    # train model, generate valid features
    with tf.Graph().as_default():

        # define model
        vgg = Join(dropout=train_flags.dropout)

        # define input data
        refs_batch, poss_batch, negs_batch, train_orders_batch = vgg.train_batch_inputs(
            train_flags.dataset_train_csv_file_path,
            train_flags.train_batch_size, train_flags.random_train_input_flag)
        train_batch = tf.concat([refs_batch, poss_batch, negs_batch], 0)

        # build model
        with tf.variable_scope(tf.get_variable_scope()):
            vgg.build(train_batch, train_test_mode=tf.constant(True, tf.bool))
            # distance_p_g = vgg.get_predict_dist(train_flags.lambda_)
            # loss
            loss = _model_loss(vgg)

        # Create an optimizer that performs gradient descent.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = (train_flags.num_per_epoch / train_flags.train_batch_size)
        decay_steps = int(num_batches_per_epoch * train_flags.num_epochs_per_decay)
        lr = tf.train.exponential_decay(train_flags.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        train_flags.learning_rate_decay_factor,
                                        staircase=True)
        vars_to_optimize = [v for v in tf.trainable_variables() if (('fc' in v.name) | ('conv' in v.name))]
        print '\nvariables to optimize'
        for v in vars_to_optimize:
            print v.name, v.get_shape().as_list()
            tf.summary.histogram(v.name, v)
        opt = tf.train.AdamOptimizer(lr)

        # grads = opt.compute_gradients(loss, var_list=vars_to_optimize)
        grads = opt.compute_gradients(loss, var_list=vars_to_optimize)
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
            _, loss_value = sess.run([train_op, loss])
            # print pool2_0.shape
            # _, loss_value, feature = sess.run([train_op, loss_mean, vgg.output], feed_dict={train_mode: True, gallery_mode: True})
            # print('feature abs mean: %s' % (np.mean(np.abs(feature))))
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = train_flags.train_batch_size / float(duration)
                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))

            if step % 100 == 0:
                summary_str, s_feature, c_feature = sess.run([summary_op, vgg.s_feature, vgg.c_feature])
                print('s_feature abs mean: %s, c_feature abs mean: %s' % (
                np.mean(np.abs(s_feature)), np.mean(np.abs(c_feature))))
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 5000 == 0 or (step + 1) == train_flags.max_step:
                # Save the model checkpoint periodically.
                checkpoint_path = os.path.join(train_flags.output_check_point_path, train_flags.checkpoint_name)
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def generate_distance(predict_flag):
    print('generate_distance(predict_flag=%s)' % (predict_flag))
    # generate predict features
    with tf.Graph().as_default():
        # build a VGG19 class object
        vgg = Join()

        # define parameter
        if predict_flag:
            gallery_path = train_flags.dataset_predict_gallery_csv_file_path
            probe_path = train_flags.dataset_predict_repeat_probe_csv_file_path
            distance_npy_name = 'predict_distance.npy'
            gallery_num = train_flags.predict_gallery_num
            probe_num = train_flags.predict_probe_num
        else:
            gallery_path = train_flags.dataset_valid_gallery_csv_file_path
            probe_path = train_flags.dataset_valid_repeat_probe_csv_file_path
            distance_npy_name = 'valid_distance.npy'
            gallery_num = train_flags.valid_gallery_num
            probe_num = train_flags.valid_probe_num

        gallery_batch, gallery_label, gallery_order = vgg.test_batch_inputs(gallery_path, train_flags.test_batch_size)
        probe_batch, probe_label, probe_order = vgg.test_batch_inputs(probe_path, train_flags.test_batch_size)
        input_batch = tf.concat([probe_batch, gallery_batch], 0, 'concat_pg')

        # build model
        with tf.variable_scope(tf.get_variable_scope()):
            vgg.build(input_batch, train_test_mode=tf.constant(False, tf.bool))
            distance_p_g = vgg.get_predict_dist(train_flags.lambda_)

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

            print('start generate distance')
            # compute distance
            dist_mat = np.zeros((gallery_num, probe_num), dtype=float)
            dist_num = gallery_num * probe_num
            print('dist_num: %s * %s = %s' % (gallery_num, probe_num, dist_num))
            batch_len = train_flags.test_batch_size
            for batch_index in range(0, dist_num, batch_len):
                batch_pg_dist, batch_porder, batch_gorder = sess.run(
                    [distance_p_g, probe_order, gallery_order])
                for i in range(batch_len):
                    dist_mat[int(batch_gorder[i])][int(batch_porder[i])] = batch_pg_dist[i][0]
                print('valid_batch_index: ' + str(batch_index) + '-' + str(batch_index + batch_len))

            # save distance
            distance_npy_path = os.path.join(train_flags.output_test_distance_path, distance_npy_name)
            np.save(distance_npy_path, dist_mat)

            coord.request_stop()
            coord.join(threads)


def valid_result():
    generate_distance(False)
    dist_mat = np.load('result/distance/valid_distance.npy')
    with open(train_flags.dataset_valid_gallery_csv_file_path, 'rb') as f:
        g_labels = np.array([int(row[1]) for row in csv.reader(f)])
    print g_labels
    with open(train_flags.dataset_valid_probe_csv_file_path, 'rb') as f:
        p_labels = np.array([int(row[1]) for row in csv.reader(f)])
    print p_labels
    map1, map2 = mAP(dist_mat, glabels=g_labels, plabels=p_labels, top_n=200)
    print('valid map: %f, %f ' % (map1, map2))


if __name__ == '__main__':
    train(retain_flag=True, start_step=1)
    valid_result()
