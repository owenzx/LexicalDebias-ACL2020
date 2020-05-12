# Towards Robustifying NLI Models Against Lexical Dataset Biases

This is the official repo for the following paper

* Towards Robustifying NLI Models Against Lexical Dataset Biases, Xiang Zhou and Mohit Bansal, ACL 2020 ([arxiv](https://arxiv.org/abs/2005.04732))

# Dependencies
This code require Python 3.4 and TensorFlow 1.12.0

# Datasets

All the datasets (train/eval) can be downloaded at [here](https://drive.google.com/file/d/1-fegPnPjL3sD6JVY7Aw158RsKDODjEwj/view?usp=sharing). For detailed description of the datasets, please check the README in the downloaded file.

# Prepare

1. Download the datasets and put it under the `data` folder.
2. Download the [GloVe embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and put it under the `data` folder.


# Usage

## Example scripts for BoW Sub-Model Orthogonality with HEX

1. First train the baseline BiLSTM model by running

```
bash scripts/baseline.sh
```

2. Train the debiased model by running

```
bash scripts/hex.sh
```

The HEX implementation is adapted from [https://github.com/HaohanWang/HEX](https://github.com/HaohanWang/HEX).

##  Evaluation
The evaluation scripts is at `evaluation.py`. When running evaluation, first change the `TESTING_DATASETS` in the file. Then run `python evaluation.py scripts/TRAININGSCRIPT`. This script will automatically generate and runs the testing scripts with respect to your training script.


More codes, model checkpoints and documentations will come soon.
