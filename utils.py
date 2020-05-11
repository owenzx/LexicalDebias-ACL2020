""" Utility functions. """
import os
import random
import tensorflow as tf
import numpy as np
import time

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
from data_processing import three2twoway_map
from shutil import copyfile

FLAGS = flags.FLAGS

def flatten_list(l):
    # only flatten one layer of nested list
    result = []
    for x in l:
        if type(x) is list:
            result.extend(x)
        else:
            result.append(x)
    return result

def normalize(inp, activation, reuse, scope):
    #BE CAUCIOUS: here reuse is set to tf.AUTO_REUSE instead of reuse
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=tf.AUTO_REUSE, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=tf.AUTO_REUSE, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def seq_xent(pred, label, mask):
    """pred:  [batch_size, seq_len, vocab],
       label: [batch_size, seq_len],
       mask:  [batch_size, seq_len]"""
    mask = tf.cast(mask, pred.dtype)
    return tf.contrib.seq2seq.sequence_loss(logits=pred, targets=label, weights=mask, average_across_timesteps=True, average_across_batch=True, softmax_loss_function=xent_onehot)

def get_bi_label(label):
    batch_size = tf.shape(label)[0]
    one_step_zero = tf.zeros((batch_size, 1), label.dtype)
    fw_label = tf.concat([label[:,1:], one_step_zero],axis=-1)
    bw_label = tf.concat([one_step_zero, label[:,:-1]], axis=-1)
    return fw_label, bw_label

def bi_seq_xent(pred, label, mask):
    pred_fw, pred_bw = pred
    mask_fw, mask_bw = mask
    label_fw, label_bw = get_bi_label(label)
    return seq_xent(pred_fw, label_fw, mask_fw) + seq_xent(pred_bw, label_bw, mask_bw)

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

def xent_onehot(logits, labels):
    class_num = tf.shape(logits)[-1]
    labels = tf.one_hot(labels, class_num)
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) / FLAGS.update_batch_size


def build_cells(cells, dim_input):
    for i,c in enumerate(cells):
        fake_inp = tf.ones((2,dim_input[i]))
        c.build(fake_inp.shape)

def get_static_pad_batch(a, batch_size, max_len):
    padded = []
    pad_len = max_len
    for i in range(len(a)//batch_size):
        batch = np.zeros
        batch = np.zeros((batch_size,pad_len), dtype=np.int32)
        for j in range(batch_size):
            batch[j,:len(a[i*batch_size+j])] = a[i*batch_size + j]
        padded.append(batch)
    return padded

def get_pad_batch(a, batch_size):
    padded = []
    for i in range(len(a)//batch_size):
        batch_len = max([len(x) for x in a[i*batch_size:(i+1)*batch_size]])
        batch = np.zeros((batch_size,batch_len), dtype=np.int32)
        for j in range(batch_size):

            batch[j,:len(a[i*batch_size+j])] = a[i*batch_size + j]
        padded.append(batch)
    return padded


def get_batch_labels(a, batch_size):
    result = []
    for i in range(len(a)//batch_size):
        batch = np.stack(a[i*batch_size:(i+1)*batch_size])
        result.append(batch)
    return result

def convert_to_stop_grad_dict(grads, theta):
    stop_grads = [tf.stop_gradient(g) if g is not None else None for g in grads]
    grads_dict = dict(zip(theta.keys(), stop_grads))
    return grads_dict
    

def convert_list_to_tensor(l):
    """Used to convert the format for static rnn to the format for dynamic rnn"""    
    t = tf.convert_to_tensor(l)
    t = tf.transpose(t, [1,0,2])
    return t

def convert_tensor_to_list(t, max_len):
    """Used to convert the format for dynamic rnn to the format for static rnn"""
    t = tf.transpose(t, [1,0,2])
    l = tf.unstack(t, num=max_len)
    return l

def get_value_summary(tag, value, writer, itr):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=tag, simple_value=value), 
    ])
    writer.add_summary(summary, itr)

def concat_all_batches(lists):
    result = np.vstack(lists).flatten().tolist()
    return result


def get_real_acc(raw_acc, output, label, twoway):
    if not twoway:
        return raw_acc
    else:
        output = np.array([three2twoway_map[x] for x in output])
        label = np.array([three2twoway_map[x] for x in label])
        acc = np.sum(output==label)/len(output)
        return acc


def backup_best_model(logdir, best_path):
    backup_path = logdir + '/pretrain_model_BEST'
    copyfile(best_path+'.index', backup_path + '.index')
    copyfile(best_path+'.meta', backup_path + '.meta')
    copyfile(best_path+'.data-00000-of-00001', backup_path + '.data-00000-of-00001')



def get_baseline_var_list():
    all_params =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var_list =  [v for v in all_params if ('bow' not in v.name) and ("hex" not in v.name)]
    print([v.name for v in var_list])
    return var_list


def get_exp_string(params):
    #generate exp_string 
    exp_string = ''
    if params["task"] == 'nli':
        exp_string += 'nli'
    else:
        raise NotImplementedError
    exp_string += 'bs'+str(params['batch_size'])  
    exp_string += 'lr' + str(params['lr'])
    exp_string += 'hidden' + str(params["hidden_dim"])
    exp_string += 'num_layers' + str(params["num_layers"])
    exp_string += 'dropout' + str(params["dropout_rate"])
    exp_string += 'model' + str(params["model_type"])
    if "neg_reg" in params.keys():
        exp_string += 'neg_reg' + str(params["neg_reg"])
    return exp_string


def get_variable_collections(scopes):
    for i, scope in enumerate(scopes):
        if i == 0:
            results = get_variable_collection(scope)
        else:
            results += get_variable_collection(scope)
    return results

def get_variable_collection(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def get_new_numbers(preds, labels, major_class):
    preds = np.array(preds)
    labels = np.array(labels)
    data_major_per = np.sum(labels==major_class)/len(preds)
    print(data_major_per)
    pred_major_per = np.sum(preds==major_class)/len(preds)
    print(pred_major_per)
    acc_major =  np.sum(np.logical_and(preds==labels, preds==major_class)) / np.sum(labels==major_class)
    acc_nonmajor =  np.sum(np.logical_and(preds==labels, preds!=major_class)) / np.sum(labels!=major_class)


    print_str = 'ACC_NONMAJOR: ' + str(acc_nonmajor) + ', ACC_MAJOR: ' + str(acc_major) 
    print(print_str)


    return data_major_per, pred_major_per, acc_major, acc_nonmajor




def restore_most(path, sess):
    cur_vars = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    cur_var_list = [v.name for v in cur_vars]
    load_vars = tf.train.list_variables(path)
    load_var_list = [v[0] for v in load_vars]
    variables_can_be_restored = [v_name for v_name in cur_var_list if v_name.split(':')[0] in load_var_list]
    var_list = [v for v in cur_vars if v.name in variables_can_be_restored]
    most_saver = tf.train.Saver(var_list = var_list)
    most_saver.restore(sess,path)



