import random

import numpy as np
import tensorbayes as tb
from termcolor import colored

from utils import load_file


def update_dict(M, feed_dict, session_file_index, session_vector_file, session_dict, item_vector, impressions_dict,
                positive_sample_dict):
    # session_index = random.randint(0, 91000)
    # session_vector_file_name = 'session_vector_' + str(int(session_index / 910))
    # session_vector_file = load_file(session_vector_file_name)
    # vector_index = session_index % 910
    while True:
        vector_index = random.randint(0, 909)
        session_index = 910 * session_file_index + vector_index
        session_id = session_dict[session_index]
        if (session_id in impressions_dict) and (session_id in positive_sample_dict):
            break

    session_vector = session_vector_file[vector_index]
    session_vector = np.transpose(np.expand_dims(session_vector, 1))
    # print(session_vector.shape)
    # print(session_index)
    # print(session_vector_file_name)
    # print(vector_index)
    # print(session_vector)

    # session_dict = load_file('session_dict')
    # session_id = session_dict[session_index]
    # print(len(session_dict))
    # print(session_id)
    impressions_vector = np.zeros((25, 157))
    # item_vector = load_file('item_vector')
    # impressions_dict = load_file('impressions_dict')
    for i in range(len(impressions_dict[session_id])):
        if int(impressions_dict[session_id][i]) in item_vector:
            impressions_vector[i] = item_vector[int(impressions_dict[session_id][i])]
    # print(impressions_vector.shape)
    # print(impressions_vector)
    # print(impressions_dict[session_id])
    # positive_sample_dict = load_file('positive_sample_dict')
    positive_sample_vector = np.zeros(157)
    if int(positive_sample_dict[session_id]) in item_vector:
        positive_sample_vector = item_vector[int(positive_sample_dict[session_id])]
    # if np.equal(positive_sample_vector, np.zeros(157)).all():
    #     raise ValueError
    positive_sample_vector = np.transpose(np.expand_dims(positive_sample_vector, 1))
    # print(positive_sample_vector.shape)
    # print(positive_sample_vector)
    # print(positive_sample_dict[session_id])
    # raise ValueError
    feed_dict.update({M.sv: session_vector, M.iv: impressions_vector, M.pv: positive_sample_vector})


def train(M, FLAGS):
    print(colored("Training is started.", "blue"))

    iterep = 1000
    n_epoch = FLAGS.epoch
    epoch = 0
    feed_dict = {}

    session_file_index = random.randint(0, 99)
    session_vector_file_name = 'session_vector_' + str(session_file_index)
    session_vector_file = load_file(session_vector_file_name)

    session_dict = load_file('session_dict')
    item_vector = load_file('item_vector')
    impressions_dict = load_file('impressions_dict')
    positive_sample_dict = load_file('positive_sample_dict')

    print(f"Iterep: {iterep}")
    print(f"Total iterations: {n_epoch * iterep}")

    for i in range(n_epoch * iterep):
        update_dict(M, feed_dict, session_file_index, session_vector_file, session_dict, item_vector, impressions_dict,
                    positive_sample_dict)
        _ = M.sess.run(M.optimizer, feed_dict)

        end_epoch, epoch = tb.utils.progbar(i, iterep, message='{}/{}'.format(epoch, i), display=True)

        if end_epoch:
            print_list = M.sess.run(M.ops_print, feed_dict)

            for j, item in enumerate(print_list):
                if j % 2 == 0:
                    print_list[j] = item.decode("ascii")
                # else:
                    # print_list[j] = round(item, 5)
                    # print_list[j] = np.around(item, 5)

            print_list += ['epoch', epoch]
            print(print_list)

            session_file_index = random.randint(0, 99)
            session_vector_file_name = 'session_vector_' + str(session_file_index)
            session_vector_file = load_file(session_vector_file_name)

    print(colored("Training ended.", "blue"))
