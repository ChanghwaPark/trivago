import os
import random

import numpy as np
import tensorbayes as tb
import tensorflow as tf
from termcolor import colored

from utils import load_file, delete_existing, save_model


def update_dict(M, feed_dict, bs, session_file_index, session_vector_file, timestamp_dict):
    session_vector = np.zeros((bs, 369539))
    timestamp_matrix = np.zeros((bs))
    for i in range(bs):
        while True:
            vector_index = random.randint(0, 909)
            session_index = 910 * session_file_index + vector_index
            session_vector[i] = session_vector_file[vector_index]
            timestamp_matrix[i] = timestamp_dict[session_index]
            if timestamp_matrix[i] != 0: break
    # timestamp_matrix += 1
    feed_dict.update({M.sv: session_vector, M.ts: timestamp_matrix})


def train(M, FLAGS, saver=None, model_name=None):
    print(colored("Training is started.", "blue"))

    iterep = 1000
    itersave = 20000
    n_epoch = FLAGS.epoch
    epoch = 0
    feed_dict = {}

    # Create log directory
    log_dir = os.path.join(FLAGS.logdir, model_name)
    delete_existing(log_dir)
    train_writer = tf.summary.FileWriter(log_dir)

    # Create checkpoint directory
    if saver:
        model_dir = os.path.join(FLAGS.ckptdir, model_name)
        delete_existing(model_dir)
        os.makedirs(model_dir)

    session_file_index = random.randint(0, 49)
    session_vector_file_name = 'session_vector_' + str(session_file_index)
    session_vector_file = load_file(session_vector_file_name)

    # test_session_file_index = session_file_index
    # test_session_vector_file = session_vector_file
    test_session_file_index = random.randint(50, 99)
    test_session_vector_file_name = 'session_vector_' + str(test_session_file_index)
    test_session_vector_file = load_file(test_session_vector_file_name)
    test_session_vector = np.zeros((FLAGS.bs, 369539))
    test_timestamp_matrix = np.zeros((FLAGS.bs))

    timestamp_dict = load_file('timestamp_dict')

    print(f"Iterep: {iterep}")
    print(f"Total iterations: {n_epoch * iterep}")

    for i in range(n_epoch * iterep):
        update_dict(M, feed_dict, FLAGS.bs, session_file_index, session_vector_file, timestamp_dict)
        summary, _ = M.sess.run(M.ops, feed_dict)
        train_writer.add_summary(summary, i + 1)
        train_writer.flush()

        end_epoch, epoch = tb.utils.progbar(i, iterep, message='{}/{}'.format(epoch, i), display=True)

        if end_epoch:

            # session_file_index = random.randint(0, 49)
            # session_vector_file_name = 'session_vector_' + str(session_file_index)
            # session_vector_file = load_file(session_vector_file_name)

            # test_session_file_index = random.randint(50, 99)
            # test_session_vector_file_name = 'session_vector_' + str(test_session_file_index)
            # test_session_vector_file = load_file((test_session_vector_file_name))

            # test_session_vector = np.zeros((FLAGS.bs, 369539))
            # test_timestamp_matrix = np.zeros((FLAGS.bs))
            for j in range(FLAGS.bs):
                while True:
                    test_vector_index = random.randint(0, 909)
                    test_session_index = 910 * test_session_file_index + test_vector_index
                    test_session_vector[j] = test_session_vector_file[test_vector_index]
                    test_timestamp_matrix[j] = timestamp_dict[test_session_index]
                    if test_timestamp_matrix[j] != 0: break

            # test_timestamp_matrix += 1

            feed_dict.update({M.test_sv: test_session_vector, M.test_ts: test_timestamp_matrix})

            print_list = M.sess.run(M.ops_print, feed_dict)

            for j, item in enumerate(print_list):
                if j % 2 == 0:
                    print_list[j] = item.decode("ascii")
                # else:
                # print_list[j] = round(item, 5)
                # print_list[j] = np.around(item, 5)

            print_list += ['epoch', epoch]
            print(print_list)

        if saver and (i + 1) % itersave == 0:
            save_model(saver, M, model_dir, i + 1)

    # Saving final model
    if saver:
        save_model(saver, M, model_dir, i + 1)

    print(colored("Training ended.", "blue"))
