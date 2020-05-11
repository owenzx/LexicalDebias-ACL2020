import tensorflow as tf
import models.blocks as blocks
from constants import *

def construct_nli_loss(params, logits, y):
    # Define the cost function
    total_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    return total_cost
