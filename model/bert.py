import numpy as np
import pandas as pd
import torch
import random
from torch import nn, optim
import torch.nn.functional as F
import math
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertTokenizer, BertModel, BertConfig
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from transformers import AlbertModel, AlbertTokenizer,AlbertForSequenceClassification


max_len = 32
seed = 666
batch_size = 16
learning_rate = 2e-6
weight_decay = 1e-5
epochs = 50
EARLY_STOP = True
EARLY_STOPPING_STEPS = 5



class Model(nn.Module):
    def __init__(self,args,drop=0.7):
        super().__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased', num_labels=args.way).to(args.gpu)

        self.embedding_size = self.encoder.pooler.dense.out_features #bert-base
#         self.embedding_size = encoder.pooler.out_features #albert
        self.fc = nn.Linear(self.embedding_size, args.way)
        # self.dropout = nn.Dropout(drop,training =False)
#         self.sig = nn.Sigmoid()
    def forward(self, x):
        #assert (x[:, 0, :] >= 0).all() and (x[:, 0, :] <= 768).all(), "input_ids超出范围"
        #assert (x[:, 1, :] == 0).any() or (x[:, 1, :] == 1).any(), "attention_mask值无效"
        #assert x.dtype == torch.long, "input_ids数据类型应为整数类型"
        #print("len input:", len(x[:, 0, :]))

        x = self.encoder(input_ids=x[:, 0, :], attention_mask=x[:,1, :])[0]

        x = x.mean(1)
        x = self.fc(x)
        return x