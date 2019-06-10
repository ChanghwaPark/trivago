import os
from pprint import pprint

import tensorflow as tf

from model_1a import model
from train_1a import train

# Define flag arguments
flags = tf.app.flags

flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_float('lrD', 0.95, 'Learning rate decay')
flags.DEFINE_integer('bs', 32, 'Batch size')
flags.DEFINE_integer('d', 128, 'Latent vector size')
flags.DEFINE_string('lf', 'bpr', 'Loss function')
flags.DEFINE_integer('epoch', 80, 'Number of epochs')
flags.DEFINE_string('logdir', 'results/logs', 'Log directory')
flags.DEFINE_string('gpu', '0', 'GPU number')

FLAGS = flags.FLAGS


def main(_):
    # Define GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    pprint(FLAGS.flag_values_dict())

    # Make main model and initialize
    M = model(FLAGS)
    M.sess.run(tf.global_variables_initializer())

    # Train the main model
    train(M, FLAGS)


if __name__ == '__main__':
    tf.app.run()
