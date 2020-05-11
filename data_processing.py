import numpy as np
import re
import random
import json
import collections
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
#import h5py
from constants import *
import os
import warnings
from copy import deepcopy
FLAGS = flags.FLAGS


LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0,
    "not_entailment": 1,
    "non-entailment": 1
}

REV_LABEL_MAP = {v:k for k,v in LABEL_MAP.items()}

three2twoway_map = {
        0:0,
        1:1,
        2:1
}

PADDING = "<PAD>"
UNKNOWN = "<UNK>"

def load_data(path):
    print("loading data from: " + path)
    return {'task':'nli', 'data':load_nli_data(path, snli=("snli" in path)), 'need_tokenize':False}
    

def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    idx = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if FLAGS.debug == True and i>=100:
                break
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            loaded_example['idx'] = idx
            idx += 1
            if 'pos_label1' in loaded_example.keys():
                if loaded_example['pos_label1'][0]=='XX' or loaded_example['pos_label2'][0]=='XX':
                    continue
            loaded_example['need_tokenize'] = False
            loaded_example['sentence1_binary_parse'] = loaded_example['token1']
            loaded_example['sentence2_binary_parse'] = loaded_example['token2']
            data.append(loaded_example)
        if FLAGS.debug == True:
            data = data[:20]
    return data


def get_rev_dict(word_indices):
    return {v:k for k,v in word_indices.items()}

def idx2tokens(idxs, rev_dict):
    # input: numpy array w/wo batch
    # output: strings
    if len(idxs.shape())==1:
        return ' '.join([rev_dict[int(i)] for i in idxs])
    elif len(idxs.shape()==2):
        results = []
        for seq in idxs:
            results.append([rev_dict[int(i)] for i in seq])
        return results

def tokenize(string):
    string = re.sub(r'\(|\)', '', string)
    return string.split()


def truncated_tokenize(string, max_len):
    string = re.sub(r'\(|\)', '', string)
    tokens = string.split()
    trunc = ' '.join(tokens[:max_len])
    return trunc

def build_dictionary(training_datasets, dict_path, params):
    """
    Extract vocabulary and build dictionary.
    """  
    warnings.warn('Start building a new dictionary.')
    word_counter = collections.Counter()
    for i, dataset in enumerate(training_datasets):
        dataset = dataset['data']
        for example in dataset:
            if "token" in example.keys():
                word_counter.update(example['token'])
            if "token1" in example.keys():
                word_counter.update(example['token1'])
                word_counter.update(example['token2'])

            else:
                word_counter.update(tokenize(example['sentence1_binary_parse']))
                word_counter.update(tokenize(example['sentence2_binary_parse']))
    word_counter = word_counter.most_common()
    vocabulary = [word[0] for word in word_counter]
    if FLAGS.max_vocab_size != None:
        vocabulary = vocabulary[:FLAGS.max_vocab_size]
    vocabulary.sort()
    vocabulary = [PADDING, UNKNOWN] + vocabulary
    print("VOCAB SIZE: %d"%len(vocabulary))
    params["vocab_size"] = len(vocabulary)
    word_indices = dict(zip(vocabulary, range(len(vocabulary))))
    with open(dict_path, 'wb') as fw:
        pickle.dump(word_indices, fw)


    return word_indices

def load_dictionary(dict_path, params):
    with open(dict_path, 'rb') as fr:
        word_indices = pickle.load(fr)
    params["vocab_size"] = len(word_indices)
    return word_indices

def get_padded_batch(word_indices, sentences, params, need_tokenize=True):
    results = [pad_sentence(word_indices, s, params, need_tokenize) for s in sentences]
    results = np.array(results)
    return results


def pad_sentence(word_indices, sentence, params, need_tokenize=True):
    result = np.zeros((params["seq_length"]), dtype=np.int32)

    if need_tokenize:
        token_sequence = tokenize(sentence)
    else:
        token_sequence = sentence
    padding = params["seq_length"] - len(token_sequence)

    for i in range(params["seq_length"]):
        if i >= len(token_sequence):
            index = word_indices[PADDING]
        else:
            if token_sequence[i] in word_indices:
                index = word_indices[token_sequence[i]]
            else:
                index = word_indices[UNKNOWN]
        result[i] = index
    return result

def pad_label(label_map, label, params):

    PAD_LABEL = 0

    result = np.zeros((params["seq_length"]), dtype=np.int32)
    token_label = label
    for i in range(params["seq_length"]):
        if i >= len(token_label):
            index = PAD_LABEL
        else:
            index = label_map[token_label[i]]
        result[i] = index
    return result


def sentences_to_padded_index_sequences(word_indices, datasets, params):
    """
    Annotate datasets with feature vectors. Adding right-sided padding. 
    """
    for i, dataset in enumerate(datasets):
        task = dataset['task']
        if 'need_tokenize' in dataset.keys():
            need_tokenize = dataset['need_tokenize']
        else:
            need_tokenize = False
        dataset = dataset['data']
        for example in dataset:
            if task == 'nli':
                for sentence in ['sentence1_binary_parse', 'sentence2_binary_parse']:
                    example[sentence + '_index_sequence'] = pad_sentence(word_indices, example[sentence], params, need_tokenize=need_tokenize)




def loadEmbedding_zeros(path, word_indices, params):
    """
    Load GloVe embeddings. Initializng OOV words to vector of zeros.
    """
    emb = np.zeros((len(word_indices), params["dim_emb"]), dtype='float32')

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if params["max_embed_to_load"] != None:
                if i >= params["max_embed_to_load"]:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[1:])

    return emb


def loadEmbedding_rand(path, word_indices):
    """
    Load GloVe embeddings. Doing a random normal initialization for OOV words.
    """
    n = len(word_indices)
    m = FLAGS.dim_emb
    emb = np.empty((n, m), dtype=np.float32)

    emb[:,:] = np.random.normal(size=(n,m))

    # Explicitly assign embedding of <PAD> to be zeros.
    emb[0:1, :] = np.zeros((1,m), dtype="float32")
    
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if FLAGS.max_embed_to_load != None:
                if i >= FLAGS.max_embed_to_load:
                    break
            
            s = line.split()
            if s[0] in word_indices:
                emb[word_indices[s[0]], :] = np.asarray(s[-m:])

    return emb

def read_pretrained_embeddings(path, word_indices):
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    n = len(word_indices)
    m = FLAGS.dim_emb
    emb = np.random.normal(size=(n,m))
    word2idx = {'PAD': PAD_TOKEN, 'BOS': BOS_TOKEN, 'EOS': EOS_TOKEN}
    weights = []
    with open(FLAGS.pretrain_embedding_path, 'r') as file:
        for index, line in enumerate(file):
            values = line.split()
            word = values[0]
            word_weights = np.asarray(values[1:], dtype = np.float32)
            word2idx[word] = index + 1
            weights.append(word_weights)
            if index + 1 == FLAGS.vocab_size:
                break
    dim_emb = len(weights[0])
    for _ in range(3):
        weights.insert(0,np.random.randn(dim_emb))

    UNK_TOKEN = len(weights)
    word2idx['UNK'] = UNK_TOKEN
    weights.append(np.random.randn(dim_emb))

    weights = np.asarray(weights, dtype = np.float32)
    
    return weights, word2idx
