import tensorflow as tf
import models.blocks as blocks

class BiLSTM(object):
    def __init__(self, params):
        ## Define hyperparameters
        self.dim = params['hidden_dim']
        self.nli_mlp_dim = params['nli_mlp_dim']
        self.sequence_length = params['seq_length']
        self.cell_type = params["cell_type"]

        self.embedding_dim = params['dim_emb']

        self.num_layers = params["num_layers"]
        self.skip_connection = params["skip_connection"]
        self.res_connection = params['res_connection']
        self.task = params['task']
        self.vocab_size = params["vocab_size"] 
        self.label_embs = False
    
    def construct_weights(self, params):
        ## Define parameters
        weights = {}
        if params['emb_on_cpu']:
            with tf.device('/cpu:0'):
                if params['embeddings'] is not None:
                    weights['E'] = tf.Variable(params['embeddings'], trainable=params['emb_train'], name='E')
                else:
                    weights['E'] = tf.Variable(tf.random_normal([params["vocab_size"], self.embedding_dim], stddev=0.1), name='E') 
        else:
            if params['embeddings'] is not None:
                weights['E'] = tf.Variable(params['embeddings'], trainable=params['emb_train'], name='E')
            else:
                weights['E'] = tf.Variable(tf.random_normal([params["vocab_size"], self.embedding_dim], stddev=0.1), name='E') 

        weights['W_mlp'] = tf.Variable(tf.random_normal([self.dim * 8, self.nli_mlp_dim], stddev=0.1), name='W_mlp')
        weights['b_mlp'] = tf.Variable(tf.random_normal([self.nli_mlp_dim], stddev=0.1), name='b_mlp')

        weights['W_cl'] = tf.Variable(tf.random_normal([self.nli_mlp_dim, 3], stddev=0.1), name='W_cl')
        weights['b_cl'] = tf.Variable(tf.random_normal([3], stddev=0.1), name='b_cl')

        weights['W_pos_mlp'] = tf.Variable(tf.random_normal([2*self.dim, self.dim], stddev=0.1), name='W_pos_mlp')
        weights['b_pos_mlp'] = tf.Variable(tf.random_normal([self.dim], stddev=0.1), name='b_pos_mlp')

        weights['W_pos_cl'] = tf.Variable(tf.random_normal([self.dim, 45], stddev=0.1), name='W_pos_cl')
        weights['b_pos_cl'] = tf.Variable(tf.random_normal([45], stddev=0.1), name='b_pos_cl')

        return weights



    def forward_model(self, inputs, weights, params, phs):
        
        keep_rate, stop_grad, zero_protect_ph = phs

        premise_x, hypothesis_x = inputs
        

        ## Function for embedding lookup and dropout at embedding layer
        def emb_drop(x):
            if params['emb_on_cpu']:
                with tf.device('/cpu:0'):
                    emb = tf.nn.embedding_lookup(weights['E'], x)
            else:
                emb = tf.nn.embedding_lookup(weights['E'], x)
            emb_drop = tf.nn.dropout(emb, keep_rate)
                
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = blocks.length(premise_x)
        hyp_seq_lengths, hyp_mask = blocks.length(hypothesis_x)

        ### BiLSTM layer ###
        premise_in = emb_drop(premise_x)
        hypothesis_in = emb_drop(hypothesis_x)
        
        results_premise_outs, results_c1 = blocks.biLSTMs(premise_in, dim=self.dim, seq_len=prem_seq_lengths, name='shared', cell_type=self.cell_type, cells=None, num_layers=self.num_layers, skip_connect=self.skip_connection, stop_grad=stop_grad, res_connect=self.res_connection, dropout_rate=0)
        results_hypothesis_outs, results_c2 = blocks.biLSTMs(hypothesis_in, dim=self.dim, seq_len=hyp_seq_lengths, name='shared', cell_type=self.cell_type, cells=None, num_layers=self.num_layers, skip_connect = self.skip_connection, stop_grad=stop_grad, res_connect=self.res_connection, dropout_rate=0)
        premise_outs = results_premise_outs[-1]
        hypothesis_outs = results_hypothesis_outs[-1]
        c1 = results_c1[-1]
        c2 = results_c2[-1]

        premise_bi = tf.concat(premise_outs, axis=2)
        hypothesis_bi = tf.concat(hypothesis_outs, axis=2)

        ### Mean pooling
        premise_sum = tf.reduce_sum(premise_bi, 1)
        premise_ave = tf.div(premise_sum, tf.expand_dims(tf.cast(prem_seq_lengths, tf.float32), -1))

        hypothesis_sum = tf.reduce_sum(hypothesis_bi, 1)
        hypothesis_ave = tf.div(hypothesis_sum, tf.expand_dims(tf.cast(hyp_seq_lengths, tf.float32), -1))

        ### Mou et al. concat layer ###
        diff = tf.subtract(premise_ave, hypothesis_ave)
        mul = tf.multiply(premise_ave, hypothesis_ave)
        h = tf.concat([premise_ave, hypothesis_ave, diff, mul], 1)

        # MLP layer
        h_mlp = tf.nn.relu(tf.matmul(h, weights['W_mlp']) + weights['b_mlp'])
        # Dropout applied to classifier
        h_drop = tf.nn.dropout(h_mlp, keep_rate)

        # Get prediction
        logits = tf.matmul(h_drop, weights['W_cl']) + weights['b_cl']

        prem_vec, hyp_vec = premise_ave, hypothesis_ave

        return prem_vec, hyp_vec, results_premise_outs, results_hypothesis_outs, logits, prem_seq_lengths, prem_mask, hyp_seq_lengths, hyp_mask, h_drop, premise_in, hypothesis_in, h_mlp 

    def _get_sen_vec_from_seq(self, seq_rep, seq_len):
        seq_bi = tf.concat(seq_rep, axis=2)
        seq_sum = tf.reduce_sum(seq_bi, 1)
        seq_ave = tf.div(seq_sum, tf.expand_dims(tf.cast(seq_len, tf.float32),-1))
        return seq_ave
