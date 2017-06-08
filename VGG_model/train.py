from datetime import datetime
import os.path, sys
import time
from tensorflow.python.platform import gfile

import numpy as np
import tensorflow as tf

import vgg19_trainable as vgg19


class Train_Flags():
    def __init__(self):

        self.current_file_path = sys.path[0]

        self.max_step = 1000000
        self.num_per_epoch = 10000
        self.num_epochs_per_decay = 30

        self.batch_size = 1

        self.initial_learning_rate = 0.0001
        self.learning_rate_decay_factor = 0.9
        self.moving_average_decay = 0.999999

        self.distance_alfa = 1

        self.output_summary_path = os.path.join(self.current_file_path, 'result','summary')
        self.output_check_point_path = os.path.join(self.current_file_path, 'result','check_point')

        self.dataset_train_csv_file_path = '/home/linze/liuhy/liuhy_github/pedestrian_retrieval/train.csv'

        self.check_path_exist()


    def check_path_exist(self):
        if not gfile.Exists(self.output_summary_path):
            gfile.MakeDirs(self.output_summary_path)
        if not gfile.Exists(self.output_check_point_path):
            gfile.MakeDirs(self.output_check_point_path)


train_flags = Train_Flags()


def _model_loss(vgg_class, refs_batch, poss_batch, negs_batch):
    # Compute the moving average of all losses

    with tf.variable_scope(tf.get_variable_scope()):
        input_batch = tf.concat([refs_batch, poss_batch, negs_batch], 0)
        vgg_class.build(input_batch)

        vgg_class.calc_loss(vgg_class.fc6, train_flags.distance_alfa)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # Compute the moving average of total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply([total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    tf.summary.scalar('loss', total_loss)

    return total_loss


def train():
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default():

        #build a VGG19 class object
        vgg = vgg19.Vgg19('./vgg19.npy')

        refs_batch, poss_batch, negs_batch = vgg.train_batch_inputs(train_flags.dataset_train_csv_file_path, train_flags.batch_size)

        loss = _model_loss(vgg, refs_batch, poss_batch, negs_batch)



        global_step = tf.get_variable('global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        num_batches_per_epoch = (train_flags.num_per_epoch / train_flags.batch_size)

        decay_steps = int(num_batches_per_epoch * train_flags.num_epochs_per_decay)

        lr = tf.train.exponential_decay(train_flags.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        train_flags.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        vars_to_optimize = [v for v in tf.trainable_variables() \
                            if v.name.startswith('fc')]
        print '\nvariables to optimize'


        for v in vars_to_optimize:
            print v.name, v.get_shape().as_list()
            tf.summary.histogram(v.name, v)

        opt = tf.train.AdamOptimizer(lr)

        grads = opt.compute_gradients(loss, var_list=vars_to_optimize)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op)

        summary_op = tf.summary.merge_all()


        sess = tf.Session()

        init = tf.global_variables_initializer()

        sess.run(init)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_writer = tf.summary.FileWriter(train_flags.output_summary_path, graph=sess.graph)
        saver = tf.train.Saver(tf.global_variables())


        print('start training')
        for step in range(train_flags.max_step):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = train_flags.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, duration))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 50000 == 0 or (step + 1) == train_flags.max_step:
                checkpoint_path = os.path.join(train_flags.output_check_point_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()


train()

