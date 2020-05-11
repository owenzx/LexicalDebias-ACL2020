""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import warnings
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_variable_collections
from constants import *

from models.bilstm_w_params import BiLSTM
from models.blocks import *
from models.losses import *
from models.hex import *
from constants import *


class SupervisedLearning:
    def __init__(self, params):
        """ must call construct_model() after initializing """
        self.lr= params["lr"]
        self.gpu_num = params["gpu_num"]

        if params["cell_type"]!="cudnn":
            warnings.warn("NOT USING CUDNN, THE MODEL WILL BE SLOWER")

        if params["neg_reg"] == 'hex':
            self.hex = HEX(params)


        if params["model_type"] == "bilstm":
            self.model = BiLSTM(params)
        else:
            raise ValueError('Unrecognized data source.')
        self.construct_weights = self.model.construct_weights
        self.forward = self.model.forward_model
        assert(params["neg_reg"] in NEG_REGS)

    def set_pretrain_embedding(self, w, word2idx):
        self.pretrain_embedding = w
        self.vocab_size = len(word2idx)

    def construct_model(self, params, input_tensors=None, prefix="mtl_"):
        self.input = input_tensors['input']
        self.label = input_tensors['label']
        self.keep_rate_ph = tf.placeholder(tf.float32, [])
        self.stop_grad_ph = tf.placeholder(tf.float32, [])
        self.zero_protect_ph = tf.placeholder(tf.bool, [])

        phs = [self.keep_rate_ph, self.stop_grad_ph, self.zero_protect_ph]

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            if 'weights' not in dir(self.model):
                weights = self.model.weights = self.model.construct_weights(params)
                if params['neg_reg']=='hex' and params['hex_share_emb'] == True:
                    self.hex.share_emb(weights['E'])
            else:
                training_scope.reuse_variables()
                weights = self.model.weights
            
            
            def get_loss(inp, label):
                nli_label = label
                prem_vec, hypo_vec, prem_hiddens, hypo_hiddens, logits, prem_len, prem_mask, hypo_len, hypo_mask, pred_vec, prem_embs, hypo_embs, pair_vec = self.forward(inp, weights, params, phs)
                self.sen_embedding_1, self.sen_embedding_2 = prem_vec, hypo_vec
                prem_mask = tf.squeeze(prem_mask, axis=[-1])
                hypo_mask = tf.squeeze(hypo_mask, axis=[-1])
                nli_loss = construct_nli_loss(params, logits, nli_label)


                #Define the additional negative regularization loss
                if params["neg_reg"] == 'none':
                    nr_loss = 0
                    nli_add_loss = 0
                elif params["neg_reg"] == 'hex':
                    prem_x, hypo_x = inp
                    hex_prem_vec, hex_hypo_vec, rep_g = self.hex.construct_hex_vec(inp, params, phs)

                    #We have to normalize these two vectors
                    rep_g = tf.nn.l2_normalize(rep_g, 0)
                    pred_vec = tf.nn.l2_normalize(pred_vec, 0)

                    pad_g = tf.zeros_like(rep_g)
                    pad_h = tf.zeros_like(pred_vec)

                    f_a = hex_classifier(pred_vec, rep_g, phs, params)
                    f_g = hex_classifier(pad_h, rep_g, phs, params)
                    f_p = hex_classifier(pred_vec, pad_g, phs, params)
                    f_l = hex_proj(f_a, f_g, params)

                    logits_g = hex_softmax(f_g, params)
                    logits_p = hex_softmax(f_p, params)
                    logits_l = hex_softmax(f_l, params)
                    if params['hex_full_test']:
                        test_logits = logits_l
                    else:
                        test_logits = logits_p

                    loss_g = construct_nli_loss(params, logits_g, nli_label)
                    loss_p = construct_nli_loss(params, logits_p, nli_label)
                    loss_l = construct_nli_loss(params, logits_l, nli_label)

                if params["neg_reg"]!='hex':
                    nli_label_loss = nli_loss + nli_add_loss

                    nli_acc = tf.contrib.metrics.accuracy(tf.argmax(logits,1), nli_label)
                else:
                    if params['hex_sup_w'] > 0:
                        nli_label_loss = loss_l + params['hex_sup_w'] * loss_g
                    else:
                        nli_label_loss = loss_l 
                    if params['hex_full_test']:
                        nli_loss_test = loss_l
                        nli_acc_test = tf.contrib.metrics.accuracy(tf.argmax(logits_l,1), nli_label)
                    else:
                        nli_loss_test = loss_p
                        nli_acc_test = tf.contrib.metrics.accuracy(tf.argmax(logits_p,1), nli_label)
                    nli_acc_train_g = tf.contrib.metrics.accuracy(tf.argmax(logits_g,1), nli_label)
                    nli_acc_train_l = tf.contrib.metrics.accuracy(tf.argmax(logits_l,1), nli_label)
                
                if params["neg_reg"]!= 'hex':
                    return logits, nli_loss, nli_acc, nli_add_loss, nli_label_loss
                elif params['neg_reg']=='hex':
                    return test_logits, nli_label_loss, loss_g, loss_l, nli_acc_train_g, nli_acc_train_l, nli_loss_test, nli_acc_test 
                else:
                    raise NotImplementedError


        if params['neg_reg']!='hex':
            logits, self.nli_loss, self.nli_acc, self.nli_add_loss, self.nli_label_loss = get_loss(self.input, self.label)

            self.output_prob = tf.nn.softmax(logits)
            
            self.output = tf.argmax(logits, 1)


            optimizer = tf.train.AdamOptimizer(self.lr)
            #The loss calculated using the nli labeled data
            self.train_op = optimizer.minimize(self.nli_loss)

            self.nli_add_loss = tf.Print(self.nli_add_loss, [self.nli_add_loss])

            sum_optimizer = tf.train.AdamOptimizer(self.lr)
            self.sum_train_op = sum_optimizer.minimize(self.nli_label_loss)
        else:
            test_logits, self.nli_label_loss, self.nli_loss_train_g, self.nli_loss_train_l, self.nli_acc_train_g, self.nli_acc_train_l, self.nli_loss_test, self.nli_acc_test = get_loss(self.input, self.label)

            self.output_prob = tf.nn.softmax(test_logits)
            self.output = tf.argmax(test_logits, 1)

            optimizer =  tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.nli_label_loss)
            self.nli_loss = self.nli_loss_test
            self.nli_acc = self.nli_acc_test


            first_train_vars = get_variable_collections(['hex_embed', 'hex_superficial', 'hex_classifier', 'hex_proj', 'hex_superficial_selfatt', 'hex_softmax'])
            self.train_op_1 = optimizer.minimize(self.nli_label_loss, var_list=first_train_vars)
            second_train_vars = get_variable_collections(['hex_classifier', 'hex_proj', 'hex_superficial_selfatt', 'hex_softmax'])
            self.train_op_2 = optimizer.minimize(self.nli_label_loss, var_list=second_train_vars)
