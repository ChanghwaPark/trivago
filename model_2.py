import tensorbayes as tb
import tensorflow as tf
from tensorbayes.layers import placeholder, dense
from termcolor import colored


def model(FLAGS):
    print(colored("Model is called.", "blue"))

    T = tb.utils.TensorDict(dict(
        sess=tf.Session(config=tb.growth_config()),
        sv=placeholder((FLAGS.bs, 369539)),
        ts=placeholder((FLAGS.bs, )),
        test_sv = placeholder((FLAGS.bs, 369539)),
        test_ts = placeholder((FLAGS.bs,))

    ))

    # h = dense(T.sv, FLAGS.d, scope='hidden', bn=False, phase=True, reuse=tf.AUTO_REUSE)
    # o = dense(h, 1, scope='out', bn=False, phase=True, reuse=tf.AUTO_REUSE)
    # test_h = dense(T.test_sv, FLAGS.d, scope='hidden', bn=False, phase=False, reuse=tf.AUTO_REUSE)
    # test_o = dense(test_h, 1, scope='out', bn=False, phase=False, reuse=tf.AUTO_REUSE)

    hidden1 = tf.get_variable("hidden1", [369539, FLAGS.d])
    hidden2 = tf.get_variable("hidden2", [FLAGS.d, 1])

    h = tf.matmul(T.sv, hidden1)
    o = tf.matmul(h, hidden2)

    test_h = tf.matmul(T.test_sv, hidden1)
    test_o = tf.matmul(test_h, hidden2)

    loss = tf.reduce_mean(tf.squared_difference(o, T.ts))

    test_o_mean = tf.reduce_mean(test_o)
    test_ts_mean = tf.reduce_mean(T.test_ts)
    test_error = tf.reduce_mean((test_o-T.test_ts)/T.test_ts) * 100.0
    error = tf.reduce_mean((o - T.ts) / T.ts) * 100.0
    # optimizer = tf.train.AdagradOptimizer(FLAGS.lr).minimize(loss)
    optimizer = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

    summary = [tf.summary.scalar('loss', loss),
               tf.summary.scalar('error', error)]
    summary = tf.summary.merge(summary)

    c = tf.constant
    T.ops_print = [c('loss'), loss,
                   c('error'), error,
                   c('test_error'), test_error,
                   c('test_o_mean'), test_o_mean,
                   c('test_ts_mean'), test_ts_mean]

    T.ops = [summary, optimizer]

    print(colored("Model is initialized.", "blue"))

    return T
