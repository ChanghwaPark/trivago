import tensorbayes as tb
import tensorflow as tf
from tensorbayes.layers import placeholder
from termcolor import colored


def get_y_hat(sv, iv, sess_q, item_p, sess_weight, item_weight):
    return tf.matmul(tf.matmul(sv, sess_q), tf.transpose(tf.matmul(iv, item_p))) \
           + tf.matmul(sv, sess_weight) + tf.matmul(iv, item_weight)


def get_loss(lf, positive_y_hat, negative_y_hat):
    if lf == 'bpr':
        return get_bpr_loss(positive_y_hat, negative_y_hat)
    elif lf == 'top':
        return get_top_loss(positive_y_hat, negative_y_hat)
    else:
        raise ValueError(f"Loss function value error: {lf}")


def get_bpr_loss(py, ny):
    return -tf.reduce_mean(tf.log_sigmoid(py - ny))


def get_top_loss(py, ny):
    return tf.reduce_mean(tf.sigmoid(ny - py) + tf.sigmoid(tf.square(ny)))


def model(FLAGS):
    print(colored("Model is called.", "blue"))

    T = tb.utils.TensorDict(dict(
        sess=tf.Session(config=tb.growth_config()),
        sv=placeholder((1, 369539)),
        iv=placeholder((25, 157)),
        pv=placeholder((1, 157))
    ))

    sess_q = tf.get_variable("sess_q", [369539, FLAGS.d])
    item_p = tf.get_variable("item_p", [157, FLAGS.d])
    sess_weight = tf.get_variable("sess_weight", [369539, 1])
    item_weight = tf.get_variable("item_weight", [157, 1])

    positive_y_hat = get_y_hat(T.sv, T.pv, sess_q, item_p, sess_weight, item_weight)
    # negative_y_hat = tf.zeros((25))
    negative_y_hat = tf.Variable(tf.zeros((25)))

    for i in range(25):
        # negative_y_hat[i] = get_y_hat(T.sv, tf.transpose(tf.expand_dims(T.iv[i], 1)), sess_q, item_p, sess_weight, item_weight)
        negative_y_hat[i].assign(get_y_hat(T.sv, tf.transpose(tf.expand_dims(T.iv[i], 1)), sess_q, item_p, sess_weight,
                                      item_weight))
    T.loss = loss = get_loss(FLAGS.lf, positive_y_hat, negative_y_hat)

    T.optimizer = tf.train.AdagradOptimizer(FLAGS.lr).minimize(loss)

    c = tf.constant
    T.ops_print = [c('loss'), loss]

    print(colored("Model is initialized.", "blue"))

    return T
