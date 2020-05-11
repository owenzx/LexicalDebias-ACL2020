import tensorflow as tf


def p_inv(matrix):
    
    """Returns the Moore-Penrose pseudo-inverse"""

    s, u, v = tf.svd(matrix)
    
    threshold = tf.reduce_max(s) * 1e-5
    s_mask = tf.boolean_mask(s, s > threshold)
    s_inv = tf.diag(tf.concat([1. / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))

    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))