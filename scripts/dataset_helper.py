import pandas as pd
import numpy as np
import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle
import random
import pdb
from torch.utils.data import DataLoader
import global_variables
import os

import unicodedata
import string
import re
import random

from torch import optim
import time

"""
This .py file contains the following classes/functions:

Classes:
    Vietnamese(df, val = False)
    Lang(name, minimum_count = 5)

Functions:
    normalizeEnString(s)
    normalizeViString(s)
    read_dataset(file, lang = None)
    split(df)
    token2index_dataset(df,en_lang,vi_lang)
    train_val_load(MAX_LEN, old_lang_obj, path)
    vocab_collate_func(batch)
    vocab_collate_func_val(batch)

"""

SOS_token = global_variables.SOS_token
EOS_token = global_variables.EOS_token
UNK_IDX = global_variables.UNK_IDX
PAD_IDX = global_variables.PAD_IDX

other_punctuations = string.punctuation.replace('!','').replace('.','').replace('?','').replace(',','').replace('-','')

class Vietnamese(Dataset):
    def __init__(self, df, val = False):
        self.df = df
        self.val = val

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        english = self.df.iloc[idx,:]['en_idized']
        viet = self.df.iloc[idx,:]['vi_idized']
        en_len = self.df.iloc[idx,:]['en_len']
        vi_len = self.df.iloc[idx,:]['vi_len']
        if self.val:
            en_data = self.df.iloc[idx,:]['en_data'].lower()
            return [viet,english,vi_len,en_len,en_data]
        else:
            return [viet,english,vi_len,en_len]


def normalizeEnString(s):
    '''
    format/clean english string
    '''
    # remove apostrophe, quote html
    s = s.replace("&apos", "").replace("&quot","")
    # remove anything thats not alphanumeric and punctuation
    s = re.sub(r"[^a-zA-Z,.!?0-9]+", r" ", s)
    # remove spacing
    s = re.sub( '\s+', ' ', s).strip()
    return s


def normalizeViString(s):
    '''
    format/clean viet string
    '''
    # remove apostrophe, quote html
    s = s.replace("&apos", "").replace("&quot","").replace("_","").replace('-','')
    # remove punctuation
    s = re.sub(r'[{}]'.format(other_punctuations), '', s)
    # remove spacing
    s = re.sub( '\s+', ' ', s).strip()
    return s


def read_dataset(file, lang = None):
    f = open(file)
    list_l = []
    for line in f:
        line = normalizeEnString(line) if lang == 'en' else normalizeViString(line)
        list_l.append(line.strip())
    df = pd.DataFrame()
    df['data'] = list_l
    return df


class Lang:
    def __init__(self, name, minimum_count = 5):
        self.name = name
        self.word2index = {}
        self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS", 2:"UKN",3:"PAD"}
        self.index2word = ["SOS","EOS","UKN","PAD"]
        self.n_words = 4  # Count SOS and EOS
        self.minimum_count = minimum_count

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word.lower())
#             if word not in string.punctuation:
#                 self.addWord(word.lower())

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
        if self.word2count[word] >= self.minimum_count:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
    #             self.index2word[self.n_words] = word
                self.index2word.append(word)
                self.n_words += 1


def split(df):
    df['en_tokenized'] = df["en_data"].apply(lambda x:x.lower().split( ))
    df['vi_tokenized'] = df['vi_data'].apply(lambda x:x.lower().split( ))
    return df



def token2index_dataset(df,en_lang,vi_lang):
    for lan in ['en','vi']:
        indices_data = []
        if lan=='en':
            lang_obj = en_lang
        else:
            lang_obj = vi_lang
        for tokens in df[lan+'_tokenized']:
            index_list = [lang_obj.word2index[token] if token in lang_obj.word2index else UNK_IDX for token in tokens]
            index_list.append(EOS_token)
#             index_list.insert(0,SOS_token)
            indices_data.append(index_list)
        df[lan+'_idized'] = indices_data
    return df


def train_val_load(MAX_LEN, old_lang_obj, path):
    en_train = read_dataset(path+"/train.tok.en", lang = 'en')
    en_val = read_dataset(path+"/dev.tok.en", lang = 'en')
    en_test = read_dataset(path+"/test.tok.en", lang = 'en')

    vi_train = read_dataset(path+"/train.tok.vi", lang = 'vi')
    vi_val = read_dataset(path+"/dev.tok.vi", lang = 'vi')
    vi_test = read_dataset(path+"/test.tok.vi", lang = 'vi')

    train = pd.DataFrame()
    train['en_data'] = en_train['data']
    train['vi_data'] = vi_train['data']

    val = pd.DataFrame()
    val['en_data'] = en_val['data']
    val['vi_data'] = vi_val['data']

    test = pd.DataFrame()
    test['en_data'] = en_test['data']
    test['vi_data'] = vi_test['data']

    if old_lang_obj:
        with open(old_lang_obj,'rb') as f:
            en_lang = pickle.load(f)
            vi_lang = pickle.load(f)
    else:
        en_lang = Lang("en")
        for ex in train['en_data']:
            en_lang.addSentence(ex)

        vi_lang = Lang("vi")
        for ex in train['vi_data']:
            vi_lang.addSentence(ex)

        with open("lang_obj.pkl",'wb') as f:
            pickle.dump(en_lang, f)
            pickle.dump(vi_lang, f)

    train = split(train)
    val = split(val)
    test = split(test)

    train = token2index_dataset(train,en_lang,vi_lang)
    val = token2index_dataset(val,en_lang,vi_lang)
    test = token2index_dataset(test,en_lang,vi_lang)

    train['en_len'] = train['en_idized'].apply(lambda x: len(x))
    train['vi_len'] = train['vi_idized'].apply(lambda x:len(x))

    val['en_len'] = val['en_idized'].apply(lambda x: len(x))
    val['vi_len'] = val['vi_idized'].apply(lambda x: len(x))

    test['en_len'] = test['en_idized'].apply(lambda x: len(x))
    test['vi_len'] = test['vi_idized'].apply(lambda x: len(x))

#     train = train[np.logical_and(train['en_len']>=2,train['vi_len']>=2)]
#     train = train[train['vi_len']<=MAX_LEN]

#     val = val[np.logical_and(val['en_len']>=2,val['vi_len']>=2)]
#     val = val[val['vi_len']<=MAX_LEN]

    return train,val,test, en_lang,vi_lang


def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """

    MAX_LEN_EN = 48
    MAX_LEN_VI = 48
    en_data = []
    vi_data = []
    en_len = []
    vi_len = []
    for datum in batch:
        en_len.append(datum[3])
        vi_len.append(datum[2])
    max_batch_length_en = max(en_len)
    max_batch_length_vi = max(vi_len)
    if max_batch_length_en < MAX_LEN_EN:
        MAX_LEN_EN = max_batch_length_en
    if max_batch_length_vi < MAX_LEN_VI:
        MAX_LEN_VI = max_batch_length_vi
    # padding
    for datum in batch:
        if datum[2]>MAX_LEN_VI:
            padded_vec_s1 = np.array(datum[0])[:MAX_LEN_VI]
        else:
            padded_vec_s1 = np.pad(np.array(datum[0]),
                                pad_width=((0,MAX_LEN_VI - datum[2])),
                                mode="constant", constant_values=PAD_IDX)
        if datum[3]>MAX_LEN_EN:
            padded_vec_s2 = np.array(datum[1])[:MAX_LEN_EN]
        else:
            padded_vec_s2 = np.pad(np.array(datum[1]),
                                pad_width=((0,MAX_LEN_EN - datum[3])),
                                mode="constant", constant_values=PAD_IDX)
        en_data.append(padded_vec_s2)
        vi_data.append(padded_vec_s1)
    vi_data = np.array(vi_data)
    en_data = np.array(en_data)
    vi_len = np.array(vi_len)
    en_len = np.array(en_len)
#     sorted_vi_len = np.argsort(-vi_len)
#     vi_data = vi_data[sorted_vi_len]
#     en_data = en_data[sorted_vi_len]
#     vi_len = vi_len[sorted_vi_len]
#     en_len = en_len[sorted_vi_len]
#     print(en_len)
    vi_len[vi_len>MAX_LEN_VI] = MAX_LEN_VI
    en_len[en_len>MAX_LEN_EN] = MAX_LEN_EN

    return [torch.from_numpy(vi_data), torch.from_numpy(en_data),
            torch.from_numpy(vi_len), torch.from_numpy(en_len)]


def vocab_collate_func_val(batch):
    return [torch.from_numpy(np.array(batch[0][0])).unsqueeze(0), torch.from_numpy(np.array(batch[0][1])).unsqueeze(0),
            torch.from_numpy(np.array(batch[0][2])).unsqueeze(0), torch.from_numpy(np.array(batch[0][3])).unsqueeze(0),batch[0][4]]
