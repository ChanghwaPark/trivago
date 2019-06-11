import os
from pprint import pprint

import tensorflow as tf

from model_2 import model
from train_2 import train

# Define flag arguments
flags = tf.app.flags

flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('bs', 32, 'Batch size')
flags.DEFINE_integer('d', 512, 'Latent vector size')
flags.DEFINE_integer('epoch', 80, 'Number of epochs')
flags.DEFINE_string('logdir', 'results/logs', 'Log directory')
flags.DEFINE_string('ckptdir', 'results/checkpoints', 'Checkpoint directory')
flags.DEFINE_string('gpu', '0', 'GPU number')

FLAGS = flags.FLAGS


def main(_):
    # Define GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    pprint(FLAGS.flag_values_dict())

    model_name = '2_'+str(FLAGS.d)+'_'+str(FLAGS.lr)
    print(f"Model name: {model_name}")

    # Make main model and initialize
    M = model(FLAGS)
    M.sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    # Train the main model
    train(M, FLAGS, saver=saver, model_name=model_name)


if __name__ == '__main__':
    tf.app.run()
