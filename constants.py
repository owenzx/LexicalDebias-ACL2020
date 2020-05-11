NLP_DATASETS = ['MNLI']
TASKS = ['nli']
NEG_REGS = ['none', 'hex']
NLI_DATASETS=["MNLI"]

DATASET_PATH = './data/'

PAD_TOKEN = 0
UNK_TOKEN = 1

VERY_NEGATIVE_NUMBER = -1e29

DATASET_PATHS = {
    "MNLI_train":"./data/train/mnli_train_bal.jsonl",
    "MNLI_dev_match":"./data/eval/mnli_dev_m.jsonl",
    "MNLI_dev_mismatch":"./data/eval/mnli_dev_mis.jsonl",
    "cwb_enh500":'./data/train/cwb_train_mix_real_500.jsonl',
    "cwb_enh20k":'./data/train/cwb_train_mix_real_20000.jsonl',
    "cwb_enh50k":'./data/train/cwb_train_mix_real_50000.jsonl',
    "cwb_syn500":'./data/train/cwb_train_mix_synt_500.jsonl',
    "cwb_syn20k":'./data/train/cwb_train_mix_synt_20000.jsonl',
    "cwb_syn50k":'./data/train/cwb_train_mix_synt_50000.jsonl',
    "cwb_syn_test_m":'./data/eval/cwb_test_m_synt.jsonl',
    "cwb_syn_test_mis":'./data/eval/cwb_test_mis_synt.jsonl',
    "cwb_enh_test":'./data/eval/cwb_test_real.jsonl',
    "wob_enh500":'./data/train/wob_train_mix_real_500.jsonl',
    "wob_enh20k":'./data/train/wob_train_mix_real_20000.jsonl',
    "wob_enh50k":'./data/train/wob_train_mix_real_50000.jsonl',
    "wob_syn500":'./data/train/wob_train_mix_synt_500.jsonl',
    "wob_syn20k":'./data/train/wob_train_mix_synt_20000.jsonl',
    "wob_syn50k":'./data/train/wob_train_mix_synt_50000.jsonl',
    "wob_syn_test_m":'./data/eval/wob_test_m_synt.jsonl',
    "wob_syn_test_mis":'./data/eval/wob_test_mis_synt.jsonl',
    "wob_enh_test":'./data/eval/wob_test_real.jsonl',
}

