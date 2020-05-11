import csv
import numpy as np
import pickle
import random
import tensorflow as tf
import os
import sys

from data_generator import DataGenerator
from sl import SupervisedLearning
from tensorflow.python.platform import flags
from constants import *

from tensorflow.python.client import timeline
from data_processing import load_data, build_dictionary, loadEmbedding_rand, sentences_to_padded_index_sequences, load_dictionary, get_rev_dict

from utils import get_exp_string, restore_most
from tqdm import tqdm
from config import params
from train_test_helper import test_on_dataset, train_hex, test_hex, biased_test, train_supervised, test_supervised


def main():
    
    train_datasets = params["train_datasets"].split(',')
    test_datasets = params["test_datasets"].split(',')
    train_paths = [DATASET_PATHS[d] for d in train_datasets]
    test_paths = [DATASET_PATHS[d] for d in test_datasets]



    tokenized_train_datasets = [load_data(p) for p in train_paths]
    tokenized_test_datasets = [load_data(p) for p in test_paths]
    if params["load_dict"] is False:
        word_indices = build_dictionary(tokenized_train_datasets, params["dict_path"], params)
    else:
        word_indices = load_dictionary(params["dict_path"], params)
    rev_dict = get_rev_dict
    pre_trained_emb = loadEmbedding_rand(params["pretrain_embedding_path"], word_indices)
    params["embeddings"] =pre_trained_emb
    sentences_to_padded_index_sequences(word_indices, tokenized_train_datasets, params)
    sentences_to_padded_index_sequences(word_indices, tokenized_test_datasets, params)
    data_generator = DataGenerator(params["batch_size"], task="nli", config=params)

    data_generator.set_datasets(tokenized_train_datasets, tokenized_test_datasets, train_datasets, test_datasets)

    if params["task"] == 'nli':
        prem_ph, hypo_ph, y_nli_ph = data_generator.get_place_holders(params["seq_length"])
        input_tensors = {'input':(prem_ph, hypo_ph), 'label':y_nli_ph}
    else:
        input_tensors = None

    sess = tf.InteractiveSession()
    
    model = SupervisedLearning(params)
    with sess.as_default():
        model.construct_model(params, input_tensors=input_tensors, prefix = "mtl_")

        saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=5)
    
        exp_string = get_exp_string(params)
    
        tf.global_variables_initializer().run()

    #load model
    resume_itr = 0
    model_file = None
    if params["resume"] or not params["train"]:
        assert(params['part_load_path']=='')
        model_file = tf.train.latest_checkpoint(params["logdir"] + '/' + exp_string)
        if params["test_iter"] > 0:
            model_file = model_file[:model_file.rindex('model')] + 'model' + str(params["test_iter"])
        elif params["test_iter"] == -4:
            # select the most recent checkpoint
            model_file = model_file
        elif params["test_iter"] == -2:
            model_file = model_file[:model_file.rindex('model')] + 'model' + "_BEST"
        elif params["test_iter"] == -3:
            model_file = params["logdir"] + '/' + 'pretrain_model_BEST'
        if model_file:
            ind1 = model_file.rindex('model')
            resume_itr = model_file[ind1+5:]
            if resume_itr != '_BEST':
                resume_itr = int(resume_itr)
            else:
                resume_itr = 0
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if params['part_load_path']!='':
        restore_most(params['part_load_path'], sess)

    if params["train"]:
        if params['neg_reg'] == 'hex':
            train_hex(params, model, saver, sess, exp_string, data_generator, resume_itr)
        else:
            train_supervised(params,model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        if params['bias_test'] == True:
            biased_test(params, model, saver, sess, exp_string, data_generator, resume_itr, params['major_class'])
        else:
            if params['neg_reg'] == 'hex':
                test_hex(params, model, saver, sess, exp_string, data_generator, resume_itr)
            else:
                test_supervised(params,model, saver, sess, exp_string, data_generator, resume_itr)

if __name__ == "__main__":
    main()
