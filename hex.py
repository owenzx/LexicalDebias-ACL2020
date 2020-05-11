import tensorflow as tf
import numpy as np
import models.blocks as blocks
from tf_p_inv import p_inv
from constants import *


def hex_proj(a, g, params):
    with tf.variable_scope("hex_proj", reuse=tf.AUTO_REUSE):
        if params['hex_final_dim'] < params['batch_size']:
            l = a - tf.matmul(tf.matmul(tf.matmul(g, p_inv(tf.matmul(g, g, transpose_a=True))),g, transpose_b=True), a) 
        else:
            small_identity = params['small_id'] * tf.eye(params['hex_final_dim'])
            l = a - tf.matmul(tf.matmul(tf.matmul(g, p_inv(tf.matmul(g, g, transpose_a=True) + small_identity)),g, transpose_b=True), a) 

    return l


def hex_classifier(h, g, phs, params):
    """Input: [h,g] or [h,0] or [0,g], Output: the layer before the linear layer of softmax"""
    with tf.variable_scope("hex_classifier", reuse=tf.AUTO_REUSE):
        keep_rate, stop_grad, _ = phs
        inp = tf.concat([h,g], -1)
        h_mlp = tf.layers.dense(inp, params['nli_mlp_dim'], tf.nn.relu)
        if params['hex_dropout']:
            h_drop = tf.nn.dropout(h_mlp, keep_rate)
        else:
            h_drop = h_mlp
        h_drop = tf.layers.dense(h_drop, params['hex_final_dim'])
    return h_drop


def hex_softmax(f, params):
    if params['final_linear']:
        with tf.variable_scope("hex_softmax", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(f, 3)
        return logits
    else:
        return f

class HEX(object):
    def __init__(self, params):
        if params['hex_share_emb'] == False:
            with tf.variable_scope("hex_embed", reuse=tf.AUTO_REUSE):
                self.embeddings = tf.Variable(params['embeddings'], trainable=params['emb_train'], name='E')
        if params['self_att']:
            self.construct_hex_vec = self.construct_hex_vec_selfatt
        else:
            self.construct_hex_vec = self.construct_hex_vec_simple
        


    def share_emb(self, embeddings):
        self.embeddings = embeddings


    def construct_hex_vec_simple(self, inputs, params, phs):
        keep_rate, stop_grad, _ = phs

        premise_x, hypothesis_x = inputs

        with tf.variable_scope("hex_superficial", reuse=tf.AUTO_REUSE):

            ## Calculate representaitons by CBOW method
            emb_premise = tf.nn.embedding_lookup(self.embeddings, premise_x) 
            emb_premise_drop = tf.nn.dropout(emb_premise, keep_rate)

            emb_hypothesis = tf.nn.embedding_lookup(self.embeddings, hypothesis_x)
            emb_hypothesis_drop = tf.nn.dropout(emb_hypothesis, keep_rate)

            premise_rep = tf.reduce_sum(emb_premise_drop, 1)
            hypothesis_rep = tf.reduce_sum(emb_hypothesis_drop, 1)

            ## Combinations
            h_diff = premise_rep - hypothesis_rep
            h_mul = premise_rep * hypothesis_rep

            ### MLP
            mlp_input = tf.concat([premise_rep, hypothesis_rep, h_diff, h_mul], 1)

            superficial_output = tf.layers.dense(mlp_input, 100)
        return premise_rep, hypothesis_rep, mlp_input

    def construct_hex_vec_selfatt(self, inputs, params, phs):
        keep_rate, stop_grad, _ = phs

        premise_x, hypothesis_x = inputs

        with tf.variable_scope("hex_superficial_selfatt", reuse=tf.AUTO_REUSE):

            emb_premise = tf.nn.embedding_lookup(self.embeddings, premise_x) 
            emb_premise_drop = tf.nn.dropout(emb_premise, keep_rate)

            emb_hypothesis = tf.nn.embedding_lookup(self.embeddings, hypothesis_x)
            emb_hypothesis_drop = tf.nn.dropout(emb_hypothesis, keep_rate)

            prem_seq_lengths, prem_mask = blocks.length(premise_x)
            hyp_seq_lengths, hyp_mask = blocks.length(hypothesis_x)

            prem_self_att= blocks.simple_self_attention_block(emb_premise_drop, params['dim_emb'], prem_seq_lengths, prem_mask, scope = 'superficial_prem_self_att')
            hypo_self_att= blocks.simple_self_attention_block(emb_hypothesis_drop, params['dim_emb'], hyp_seq_lengths, hyp_mask, scope = 'superficial_hypo_self_att')


            premise_rep = tf.reduce_sum(prem_self_att, 1)
            hypothesis_rep = tf.reduce_sum(hypo_self_att, 1)

            ## Combinations
            h_diff = premise_rep - hypothesis_rep
            h_mul = premise_rep * hypothesis_rep

            ### MLP
            mlp_input = tf.concat([premise_rep, hypothesis_rep, h_diff, h_mul], 1)
        return premise_rep, hypothesis_rep, mlp_input
