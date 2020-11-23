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

# remove punctuation
other_punctuations = string.punctuation.replace('!','').replace('.','').replace('?','').replace(',','').replace('-','')

'''
regex: https://www.regular-expressions.info/tutorial.html
'''

class Vietnamese(Dataset):
	'''
	Class that represents a train/validation/test dataset that's readable for PyTorch
	Note that this class inherits torch.utils.data.Dataset
	'''
	def __init__(self, df, val = False):
		self.df = df
		self.val = val

	def __len__(self):
		# return number of pairs, triggered when you call len(dataset)
		return len(self.df)

	def __getitem__(self, idx):
		# trigged when you call dataset[i]
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
	'''
	read dataset as pandas df and normalize each sentence
	'''
	f = open(file)
	list_l = []
	for line in f:
		line = normalizeEnString(line) if lang == 'en' else normalizeViString(line)
		list_l.append(line.strip())
	df = pd.DataFrame()
	df['data'] = list_l
	return df


class Lang:
	'''
	represent each word in language as one-hot vector.
	name: name of language
	word2index: get index from word (word --> index)
	word2count: get counts for each word in language
	index2word: unique index per word (index --> word)
	n_words: number of unique words in language
	minimum_count: threshold for word count to be added in language
	'''
	def __init__(self, name, minimum_count = 5):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = ["<sos>","<eos>","<unk>","<pad>"]
		self.n_words = 4  # Count SOS, EOS, PAD, UNK tokens
		self.minimum_count = minimum_count

	def addSentence(self, sentence):
		'''
		add sentence to lang obj
		'''
		for word in sentence.split(' '):
			self.addWord(word.lower())


	def addWord(self, word):
		'''
		add word to language obj and update data structures
		'''
		# update word count
		if word not in self.word2count:
			self.word2count[word] = 1
		else:
			self.word2count[word] += 1
		# check if the word count is greater than the minimum count
		if self.word2count[word] >= self.minimum_count:
			if word not in self.word2index:
				self.word2index[word] = self.n_words
				self.index2word.append(word)
				self.n_words += 1

def split(df):
	'''
	create tokenized col for source and target lang
	'''
	df['en_tokenized'] = df["en_data"].apply(lambda x:x.lower().split( ))
	df['vi_tokenized'] = df['vi_data'].apply(lambda x:x.lower().split( ))
	return df



def token2index_dataset(df,en_lang,vi_lang):
	'''
	create indicized column for pandas df
	'''
	for lan in ['en','vi']:
		indices_data = []
		if lan=='en':
			lang_obj = en_lang
		else:
			lang_obj = vi_lang
		for tokens in df[lan+'_tokenized']:
			index_list = [lang_obj.word2index[token] if token in lang_obj.word2index else UNK_IDX for token in tokens]
			index_list.append(EOS_token)
			indices_data.append(index_list)
		df[lan+'_idized'] = indices_data
	return df

def train_val_load(old_lang_obj, path):
	'''
	load source and target language objects
	load pandas df with len, tokenize, indicized cols for src/trg
	'''

	# load en and normalize
	en_train = read_dataset(path+"/iwslt-vi-en/train.tok.en", lang = 'en')
	en_val = read_dataset(path+"/iwslt-vi-en/dev.tok.en", lang = 'en')
	en_test = read_dataset(path+"/iwslt-vi-en/test.tok.en", lang = 'en')

	# load vn and normalize
	vi_train = read_dataset(path+"/iwslt-vi-en/train.tok.vi", lang = 'vi')
	vi_val = read_dataset(path+"/iwslt-vi-en/dev.tok.vi", lang = 'vi')
	vi_test = read_dataset(path+"/iwslt-vi-en/test.tok.vi", lang = 'vi')

	# add sentence column for train
	train = pd.DataFrame()
	train['en_data'] = en_train['data']
	train['vi_data'] = vi_train['data']

	# add sentence column for val
	val = pd.DataFrame()
	val['en_data'] = en_val['data']
	val['vi_data'] = vi_val['data']

	# add sentence column for test
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

	# get tokenized columns
	train = split(train)
	val = split(val)
	test = split(test)

	# get indicized columns
	train = token2index_dataset(train,en_lang,vi_lang)
	val = token2index_dataset(val,en_lang,vi_lang)
	test = token2index_dataset(test,en_lang,vi_lang)

	# get sentence lengths for train
	train['en_len'] = train['en_idized'].apply(lambda x: len(x))
	train['vi_len'] = train['vi_idized'].apply(lambda x:len(x))

	# get sentence lengths for val
	val['en_len'] = val['en_idized'].apply(lambda x: len(x))
	val['vi_len'] = val['vi_idized'].apply(lambda x: len(x))

	# get sentence lengths for test
	test['en_len'] = test['en_idized'].apply(lambda x: len(x))
	test['vi_len'] = test['vi_idized'].apply(lambda x: len(x))

#     only sentences that have 2 or more words
#     train = train[np.logical_and(train['en_len']>=2,train['vi_len']>=2)]
#     train = train[train['vi_len']<=MAX_LEN]

#     val = val[np.logical_and(val['en_len']>=2,val['vi_len']>=2)]
#     val = val[val['vi_len']<=MAX_LEN]
	return train, val, test, en_lang, vi_lang

def vocab_collate_func(batch, MAX_LEN):
	'''
	Customized function for DataLoader that dynamically pads the batch so that all
	data have the same length
	'''

	en_data, vi_data, en_len, vi_len = [], [], [], []
	for datum in batch:
		# add sent len for each sent in batch
		en_len.append(datum[3])
		vi_len.append(datum[2])

	# get max len and source/target len
	max_batch_length_en = max(en_len)
	max_batch_length_vi = max(vi_len)

	MAX_LEN_VI = np.min([ np.max(vi_len), MAX_LEN])
	MAX_LEN_EN = np.min([np.max(en_len), MAX_LEN])

	# clip the length to the max length
	vi_len = np.clip(vi_len, a_min = None, a_max = MAX_LEN_VI)
	en_len = np.clip(en_len, a_min = None, a_max = MAX_LEN_EN)

	# padding
	for datum in batch:
		if datum[2]>MAX_LEN_VI:
			padded_vec_s1 = np.array(datum[0])[:MAX_LEN_VI]
		else:
			padded_vec_s1 = np.pad(np.array(datum[0]), pad_width=((0,MAX_LEN_VI - datum[2])), mode="constant", constant_values=PAD_IDX)
		if datum[3]>MAX_LEN_EN:
			padded_vec_s2 = np.array(datum[1])[:MAX_LEN_EN]
		else:
			padded_vec_s2 = np.pad(np.array(datum[1]), pad_width=((0,MAX_LEN_EN - datum[3])), mode="constant", constant_values=PAD_IDX)
		en_data.append(padded_vec_s2)
		vi_data.append(padded_vec_s1)
	vi_data = np.array(vi_data)
	en_data = np.array(en_data)
	vi_len = np.array(vi_len)
	en_len = np.array(en_len)

	vi_len[vi_len>MAX_LEN_VI] = MAX_LEN_VI
	en_len[en_len>MAX_LEN_EN] = MAX_LEN_EN

	return [torch.from_numpy(vi_data), torch.from_numpy(en_data), torch.from_numpy(vi_len), torch.from_numpy(en_len)]

def vocab_collate_func_val(batch):
	'''
	Customized function for DataLoader that dynamically pads the batch so that all data have the same length (for validation)
	'''
	vi = torch.from_numpy(np.array(batch[0][0])).unsqueeze(0)
	en = torch.from_numpy(np.array(batch[0][1])).unsqueeze(0)
	vi_len = torch.from_numpy(np.array(batch[0][2])).unsqueeze(0)
	en_len = torch.from_numpy(np.array(batch[0][3])).unsqueeze(0)
	en_data = batch[0][4]
	return [vi, en, vi_len, en_len, en_data]
