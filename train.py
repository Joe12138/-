from transformers import BertTokenizer
from pathlib import Path
import torch

import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split

import datetime
import logging

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc


label_cols = ['LN'+str(i) for i in range(1, 22)]

tokenizer = BertTokenizer(vocab_file='./chinese_roberta_wwm_large_ext_pytorch/vocab.txt')

databunch = BertDataBunch(data_dir='./Data/loan/new_data',
                          label_dir='./Data/loan/new_data',
                          tokenizer=tokenizer,
                          train_file='train.csv',
                          val_file='',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=label_cols,
                          batch_size_per_gpu=4,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=True,
                          backend='nccl',
                          model_type='bert')

metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})
device_cuda = torch.device("cuda")
logger = logging.getLogger()

learner = BertLearner.from_pretrained_model(dataBunch=databunch,
                                            pretrained_path='./chinese_roberta_wwm_large_ext_pytorch',
                                            metrics=metrics,
                                            device=device_cuda,
                                            logger=logger,
                                            output_dir='./Data/loan/data/model/keda',
                                            multi_label=True)

learner.fit(epochs=6,
			lr=3e-5,
			validate=False, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")

learner.save_model()


text_list = list(pd.read_csv('./Data/loan/new_data/test.csv')['text'].values)
output = learner.predict_batch(text_list)

for ele, text in zip(output, text_list):
    print(ele)
    print(text)