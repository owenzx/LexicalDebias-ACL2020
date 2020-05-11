import tensorflow as tf
from tensorflow.python.platform import flags
import os

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('task', 'nli', 'determine what task to run, currently only support nli and senti')
flags.DEFINE_string('train_datasets', 'mnli', 'the datasets for training, use comma to saparate')
flags.DEFINE_string('test_datasets', 'snli', 'the datasets for testing')
flags.DEFINE_string('dict_path', None, 'the path to save/load the dictionary')
flags.DEFINE_bool('load_dict', True, 'if True, load the dict from dict_path')

## Training options
flags.DEFINE_integer('batch_size', 32, 'number of tasks sampled per meta-update')
flags.DEFINE_float('lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_float('dropout_rate', 0.1, 'dropout rate')
flags.DEFINE_bool('clip_grad', False, 'Whether to clip the grad of not, default range is -10 to 10')
flags.DEFINE_integer('pretrain_epochs', 0, 'number of pre-training epochs')
flags.DEFINE_integer('pretrain_itrs', 0, 'number of pre-training itrs')

flags.DEFINE_integer('seq_length', 50, 'longest sequence length, sentences longer than this value are truncated')


## Model options
flags.DEFINE_string('model_type', 'bilstm', 'choose the model used in the experiment')
flags.DEFINE_string('cell_type', 'cudnn', 'choose the rnn cell used in the model')
flags.DEFINE_integer('num_layers', 1, "number of layers in the model")
flags.DEFINE_bool('skip_connection', True, 'whether to use skip connection between stacked lstms')
flags.DEFINE_bool('res_connection', True, 'whether to use residual connection between stacked lstms')
flags.DEFINE_integer('hidden_dim', 300, "default dimension for most of the hidden layers")
flags.DEFINE_integer('dim_emb', 300, 'the dimension of the embedding')
flags.DEFINE_integer('nli_mlp_dim',800,'the dimension of the MLP layer before the classifier for nli')
flags.DEFINE_string('neg_reg', 'none', 'determine what kind of negative regularization to use in the model')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test_set', False, 'True to use test set for validation/test, False to use the validation set')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('save_result', False, 'Save the result on test_set for analysis')
flags.DEFINE_integer('main_val_idx', 0, 'The index of the validation set for selecting the best model')
flags.DEFINE_string('part_load_path', '', 'the location of the pretrained baseline model to load')

flags.DEFINE_bool('emb_train',True, 'whether to train the embedding or not')
flags.DEFINE_integer('max_vocab_size', None, 'the size of vocabulary, default set to 40000, which is a common setting for GloVe')
flags.DEFINE_integer('max_embed_to_load', None, 'max number of embedding to load from file')
flags.DEFINE_string('pretrain_embedding_path', '', 'the path of the pretrained embedding file')

flags.DEFINE_string('gpu_id', "-1", 'the id of the gpu to use')
flags.DEFINE_integer('gpu_num',1, 'the number of GPUs to use')

flags.DEFINE_bool('debug', False, 'whether run in debug mode')


#Special configs for HEX experiments
flags.DEFINE_integer('freeze_sf_itr', 99999, 'after this iteration, the parameter in the superficial model is freezed')

flags.DEFINE_bool('bias_test', False, 'whether to use the biased test')
flags.DEFINE_integer('major_class', None, 'the major class for the biased test')

flags.DEFINE_bool('all_twoway', False, 'if True, view all the tasks as a two way classification: entailment/non-entailment')
flags.DEFINE_bool('self_att', False, 'to use self_att in the superifical model')
flags.DEFINE_bool('best_loss', False, 'save the model with lowest loss on development set')
flags.DEFINE_bool('share_embedding', False, 'set to true to share embedding between superfial model and main model')
flags.DEFINE_bool('emb_on_cpu', False, 'set True to force the calucation using the embedding is on CPU (for RMSProp)')

flags.DEFINE_integer('hex_final_dim', 3, 'the final dimension of hex to tune')

flags.DEFINE_float('small_id', 1e-7, 'hex tunable small identity matrix')

flags.DEFINE_bool('final_linear', False, 'if True, add another linear layer before softmax for HEX')
flags.DEFINE_bool('hex_full_test', False, 'if True, use full to test hex')

flags.DEFINE_float('hex_sup_w', 0.0, 'if non-zero, also train the superficial model')
flags.DEFINE_bool('hex_dropout', True, 'if False, stop using dropout in hex')
flags.DEFINE_bool('hex_share_emb', False, 'if true, share embedding between main model and hex')

# get rid of lazy initialization
gpu_ids = [int(x) for x in FLAGS.gpu_id.split(',')]
print(gpu_ids)

tmp = FLAGS.__flags
params = {k:tmp[k].value for k in tmp.keys()}
print(params)
gpu_ids = [int(x) for x in params["gpu_id"].split(',')]
print(gpu_ids)
if gpu_ids[0] == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif gpu_ids[0] == -2:
    pass
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = params["gpu_id"]
    assert(params["gpu_num"]==len(gpu_ids))

if params['bias_test'] == True:
    assert(params['major_class'] is not None)
