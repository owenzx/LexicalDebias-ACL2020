"""

Functions and components that can be slotted into tensorflow models.

TODO: Write functions for various types of attention.

"""

import tensorflow as tf
from tf_utils.ops import bi_cudnn_rnn_encoder


def noam_norm(x, epsilon=1.0, scope=None, reuse=None):
    """One version of layer normalization."""
    with tf.name_scope(scope, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(tf.to_float(shape[-1]))

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer = regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret,[0,2,1,3])


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
    x: a Tensor with shape [..., a, b]
    Returns:
    a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret



def length(sequence):
    """
    Get true length of sequences (without padding), and mask for true-length in max-length.

    Input of shape: (batch_size, max_seq_length, hidden_dim)
    Output shapes, 
    length: (batch_size)
    mask: (batch_size, max_seq_length, 1)
    """
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask



def l2_blstm(a, b, lens):
    a_c = tf.concat(a, axis = 2)
    b_c = tf.concat(b, axis = 2)
    #The mask for a should be the same as the mask for b
    sum_length = tf.to_float(tf.reduce_sum(lens))
    a_c = tf.stop_gradient(a_c)
    l2_loss = tf.norm(a_c-b_c, ord='euclidean')
    return l2_loss/sum_length

def cos_blstm(a, b, lens):
    a_c = tf.concat(a, axis = 2)
    b_c = tf.concat(b, axis = 2)
    #The mask for a should be the same as the mask for b
    sum_length = tf.to_float(tf.reduce_sum(lens))
    a_c = tf.stop_gradient(a_c)
    cos_loss = tf.losses.cosine_distance(a_c, b_c)
    return cos_loss/sum_length



def biLSTM(inputs, dim, seq_len, name, cell_type="cudnn", cells=None, is_training=True, dropout_rate=0.0):
    """
    A Bi-Directional LSTM layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

    Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
    Same shape for cell states.
    """
    if cell_type=="cudnn":
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:
            hidden_states, cell_states = bi_cudnn_rnn_encoder('lstm', dim, 1, dropout_rate,  inputs, seq_len, is_training)
    else:
        with tf.name_scope(name) as scope:
            with tf.variable_scope('forward' + name) as scope:
                if cell_type == "lstm":
                    lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
            with tf.variable_scope('backward' + name) as scope:
                if cell_type == "lstm":
                    lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)
            
            with tf.variable_scope(name+'blstm', reuse=tf.AUTO_REUSE):
                hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd, cell_bw=lstm_bwd, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states

def biLSTMs(inputs, dim, seq_len, name, cell_type="cudnn", cells=None, num_layers=1, skip_connect=False, stop_grad=0, res_connect = False, dropout_rate=0.0):
    #This block implements Stacked BiLSTM with skip-connection
    #The skip_conncetion implemented here only supports the connection from the lowest level representation to the top.
    lowest_input =  inputs
    blstm_input = lowest_input
    results_hidden_states = []
    results_cell_states = []

    with tf.name_scope(name):
        for idx_l in range(num_layers):
            if idx_l == num_layers-1: 
                blstm_input = tf.stop_gradient(blstm_input) * stop_grad + blstm_input * (1-stop_grad)
            if cells !=None:
                hidden_states, cell_states = biLSTM(blstm_input, dim, seq_len, name+str(idx_l), cell_type=cell_type, cells=cells[idx_l], dropout_rate=dropout_rate)
            else:
                hidden_states, cell_states = biLSTM(blstm_input, dim, seq_len, name+str(idx_l), cell_type=cell_type, cells=None, dropout_rate=dropout_rate)
            blstm_input = tf.concat(hidden_states, axis = 2)
            if skip_connect == True:
                if res_connect == False:
                    blstm_input = tf.concat([blstm_input, lowest_input], axis=2)
                else:
                    if idx_l!=0:
                        blstm_input = blstm_input + tf.concat(results_hidden_states[-1], axis=2)
                    blstm_input = tf.concat([blstm_input, lowest_input], axis=2)
            results_hidden_states.append(hidden_states)
            results_cell_states.append(cell_states)
    return results_hidden_states, results_cell_states



def LSTM(inputs, dim, seq_len, name):
    """
    An LSTM layer. Returns hidden states and cell states as a tuple.

    Ouput shape of hidden states: (batch_size, max_seq_length, hidden_dim)
    Same shape for cell states.
    """
    with tf.name_scope(name) as scope:
        cell = tf.contrib.rnn.LSTMCell(num_units=dim)
        hidden_states, cell_states = tf.nn.dynamic_rnn(cell, inputs=inputs, sequence_length=seq_len, dtype=tf.float32, scope=name)

    return hidden_states, cell_states


def last_output(output, true_length):
    """
    To get the last hidden layer form a dynamically unrolled RNN.
    Input of shape (batch_size, max_seq_length, hidden_dim).

    true_length: Tensor of shape (batch_size). Such a tensor is given by the length() function.
    Output of shape (batch_size, hidden_dim).
    """
    max_length = int(output.get_shape()[1])
    length_mask = tf.expand_dims(tf.one_hot(true_length-1, max_length, on_value=1., off_value=0.), -1)
    last_output = tf.reduce_sum(tf.multiply(output, length_mask), 1)
    return last_output


def masked_softmax(scores, mask):
    """
    Used to calculcate a softmax score with true sequence length (without padding), rather than max-sequence length.

    Input shape: (batch_size, max_seq_length, hidden_dim). 
    mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the length() function.
    """
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights


def compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim):
    """ computes a (batch, x_word_dim, key_word_dim) bool mask for clients that want masking """
    if x_mask is None and mem_mask is None:
        return None
    elif x_mask is None or mem_mask is None:
        raise NotImplementedError()

    x_mask = tf.cast(x_mask,dtype=bool)
    mem_mask = tf.cast(tf.transpose(mem_mask,perm=[0,2,1]), dtype=bool)
    join_mask = tf.logical_and(x_mask, mem_mask)
    return join_mask


def compute_KL_divergence_from_logits(logits_p, logits_q, mask):
    """compute the kl-divergence between 2 logits vector p and q, default returns a scalar"""
    #suppose p and q are both tensors with shape [batch, seq_len, #class]
    #suppose mask is a tensor with shape [batch, seq_len]
    p = tf.nn.softmax(logits_p)
    q = tf.nn.softmax(logits_q)
    mask = tf.expand_dims(mask, -1)
    total_kl = mask*p*tf.log(p/q)
    return tf.reduce_mean(total_kl)

def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          seq_len = None,
                          mask = None,
                          is_training = True,
                          scope=None,
                          reuse = None,
                          dropout = 0.0):
    """dot-product attention.
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    is_training: a bool of training
    scope: an optional string
    Returns:
    A Tensor.
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias:
            b = tf.get_variable("bias",
                    logits.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            logits += b
        if mask is not None:
            shapes = [x  if x != None else -1 for x in logits.shape.as_list()]
            mask = tf.reshape(mask, [shapes[0],1,1,shapes[-1]])
            logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        return tf.matmul(weights, v)


def simple_self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 3, scope = "self_attention_ffn", reuse=None, is_training = True, bias = True, dropout = 0.0):
    """a simplified implementation of self-attention block without layer dropout"""
    norm_fn = layer_norm
    
    with tf.variable_scope(scope, reuse = reuse):
        outputs = norm_fn(inputs, scope = "layer_norm_1", reuse=reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs= multihead_attention(outputs, num_filters,
            num_heads = num_heads, seq_len = seq_len, reuse = reuse,
            mask = mask, is_training = is_training, bias = bias, dropout = dropout)
        residual = tf.nn.dropout(outputs, 1.0 - dropout) + inputs

        outputs = norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
        outputs = conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout) + residual
    return outputs



def self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.0, sublayers = (1, 1)):
    norm_fn = layer_norm#tf.contrib.layers.layer_norm #tf.contrib.layers.layer_norm or noam_norm

    with tf.variable_scope(scope, reuse = reuse):
        l, L = sublayers
        # Self attention
        outputs = norm_fn(inputs, scope = "layer_norm_1", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
            num_heads = num_heads, seq_len = seq_len, reuse = reuse,
            mask = mask, is_training = is_training, bias = bias, dropout = dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1
        # Feed-forward
        outputs = norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
        outputs = conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l

def multihead_attention(queries, units, num_heads,
                        memory = None,
                        seq_len = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.0):
    with tf.variable_scope(scope, reuse = reuse):
        # Self attention
        if memory is None:
            memory = queries

        memory = conv(memory, 2 * units, name = "memory_projection", reuse = reuse)
        query = conv(queries, units, name = "query_projection", reuse = reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory,2,axis = 2)]

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5
        x = dot_product_attention(Q,K,V,
                                  bias = bias,
                                  seq_len = seq_len,
                                  mask = mask,
                                  is_training = is_training,
                                  scope = "dot_product_attention",
                                  reuse = reuse, dropout = dropout)
        return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))


def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
    initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
    regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        outputs = conv_func(inputs, kernel_, strides, "VALID")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs


def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)
