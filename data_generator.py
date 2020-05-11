""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf
import h5py
from tqdm import tqdm, trange

from tensorflow.python.platform import flags
import nltk
from constants import *
FLAGS = flags.FLAGS

class DataGenerator(object):
    """
    Data Generator capable of generating batches of text data.
    """
    def __init__(self, batch_size, task, config={}):
        """
        Args:
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        if task=="nli":
            self.get_place_holders = self.get_nli_place_holders
            self.get_train_batch = self.get_sl_train_batch
        



    def preprocessed_text(self, text, word2idx):
        words = nltk.word_tokenize(text.lower())
        result = []
        for i, w in enumerate(words):
            if w in word2idx.keys():
                result.append(w)
            else:
                result.append('UNK')
        return ' '.join(result)


    def get_nli_place_holders(self, max_len):
        self.prem_ph = tf.placeholder(tf.int64, [None, max_len])
        self.hypo_ph = tf.placeholder(tf.int64, [None, max_len])
        self.nli_y_ph = tf.placeholder(tf.int64, [None])
        return self.prem_ph, self.hypo_ph, self.nli_y_ph

    def set_datasets(self, train_datasets, test_datasets, train_dataset_names, test_dataset_names):
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.num_train_datasets = len(self.train_datasets)
        self.num_test_datasets = len(self.test_datasets)
        self.train_dataset_names = []
        for name in train_dataset_names:
            self.train_dataset_names.append(name)
        self.test_dataset_names = test_dataset_names
    


    def get_sl_train_batch(self):
        dataset_sizes = [len(d['data']) for d in self.train_datasets]
        dataset_probs = [l/sum(dataset_sizes) for l in dataset_sizes]

        nli_batch_idx = 0

        while True:
            dataset_no = np.random.choice(len(self.train_datasets), p=dataset_probs)
            dataset_p = self.train_datasets[dataset_no]
            dataset_task = dataset_p['task']
            dataset = dataset_p['data']
            if dataset_task == 'nli':
                idxs = range(nli_batch_idx*self.batch_size,(nli_batch_idx+1)*self.batch_size)
                prem_vecs = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in idxs])
                hypo_vecs = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in idxs])
                labels = [dataset[i]['label'] for i in idxs]
                        
                yield {'name':'train',
                        'prem':prem_vecs,
                        'hypo':hypo_vecs,
                        'y': labels,
                        'task': 'nli'}
                nli_batch_idx = nli_batch_idx + 1
                if nli_batch_idx >= dataset_sizes[dataset_no]//self.batch_size:
                    nli_batch_idx = 0
                    np.random.seed(1)
                    np.random.shuffle(dataset)

         
    def get_test_batch(self, name):
        dataset_p = self.test_datasets[self.test_dataset_names.index(name)]
        dataset = dataset_p['data']
        dataset_task = dataset_p['task']
        if dataset_task == 'nli':
            for batch_idx in range(len(dataset)//self.batch_size+1):
                start_idx = batch_idx * self.batch_size
                if (batch_idx + 1) * self.batch_size > len(dataset):
                    if start_idx >= len(dataset):
                        break
                    end_idx = len(dataset)
                else:
                    end_idx = (batch_idx+1) * self.batch_size
                prem_vecs = np.vstack([dataset[i]['sentence1_binary_parse_index_sequence'] for i in range(start_idx, end_idx)])
                hypo_vecs = np.vstack([dataset[i]['sentence2_binary_parse_index_sequence'] for i in range(start_idx, end_idx)])
                labels = [dataset[i]['label'] for i in range(start_idx, end_idx)]
                yield {'name':name,
                        'prem':prem_vecs,
                        'hypo':hypo_vecs,
                        'y': labels,
                        'task': 'nli'}

       
    def get_test_inps(self, name):
        dataset = self.test_datasets[self.test_dataset_names.index(name)]
        inps = {'prem':[], 'hypo':[]}
        real_len = len(dataset['data'])//self.batch_size*self.batch_size
        for i in range(real_len):
            if 'token1' in dataset['data'][0].keys():
                inps['prem'].append(' '.join(dataset['data'][i]['token1']))
                inps['hypo'].append(' '.join(dataset['data'][i]['token2']))
            else:
                inps['prem'].append(' '.join(dataset['data'][i]['sentence1_binary_parse']))
                inps['hypo'].append(' '.join(dataset['data'][i]['sentence2_binary_parse']))
        return inps
