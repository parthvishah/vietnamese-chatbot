import os
import time
import torch
from torch.utils.data import DataLoader
from functools import partial
import sys
import logging as log
from datetime import datetime as dt
import time
import copy
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import global_variables
import dataset_helper
import nnet_models
from args import args, check_args

"""
This .py file contains the following functions:

	get_full_filepath(path, enc_type)
	save_models(model, path, nn_type)
	get_binned_bl_score(nmt_model, val_dataset, location, batchSize)
	showAttention(input_sentence, output_words, attentions)
	get_encoded_batch(sentence, lang_obj, use_cuda)
"""

def get_full_filepath(path, filename):
	'''
	get the full checkpoint file path
	'''
	filename = filename+".pth"
	return os.path.join(path, filename)

def save_models(model, path, fiilename):
	'''
	save the model
	'''
	if not os.path.exists(path):
		os.makedirs(path)
	filename = filename+".pth"
	torch.save(model, os.path.join(path, filename))

def get_binned_bl_score(encoder, decoder, val_dataset, attn_flag, beam_size, location, min_len = 0, max_len = 30):
	'''
	return plot for binned bleu scores
	'''

	attn_str = 'attention' if attn_flag == True else 'no_attention'

	# set bins
	len_threshold = np.arange(min_len, max_len + 1, 5)
	# intiate bleu score list
	bin_bl_score = np.zeros(len(len_threshold))

	for i in notebook.tqdm(range(1, len(len_threshold)), total = len(len_threshold)-1):
		# set lower and upper bound buckets
		lower_bound = len_threshold[i-1]
		upper_bound = len_threshold[i]

		# subset val df
		temp_dataset = copy.deepcopy(val_dataset);
		temp_dataset.df = temp_dataset.df[(temp_dataset.df['vi_len'] > lower_bound) & (temp_dataset.df['vi_len'] <= upper_bound)];

		# val dataloader
		temp_loader = DataLoader(temp_dataset, batch_size = 1, collate_fn = vocab_collate_func_val, shuffle = False, num_workers=0)

		# evaluate
		bin_bl_score[i], _, _, _ = validation_beam_search(encoder, decoder, temp_loader, en_lang, vi_lang, attn_str, beam_size, verbose = False)

	# plot bleu score vs. sent lengh
	len_threshold = len_threshold[1:]
	bin_bl_score = bin_bl_score[1:]
	fig = plt.figure()
	plt.plot(len_threshold, bin_bl_score, '+-')
	plt.ylim(0, np.max(bin_bl_score)+1)
	plt.xlabel('sentence length')
	plt.ylabel('bleu score')
	plt.title('Bleu Score vs. Sentence Length')
	fig.tight_layout()
	fig.savefig(os.path.join(location,'binned_bl_score_{}.png'.format(time.strftime("%Y%m%d-%H.%M.%S"))))

	return len_threshold, bin_bl_score, fig


# def showAttention(input_sentence, output_words, attentions):
# 	# Set up figure with colorbar
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	cax = ax.matshow(attentions, cmap='bone', aspect='auto')
# 	fig.colorbar(cax)
#
# 	# Set up axes
# 	ax.set_xticklabels([''] + input_sentence.split(' ') + [global_variables.EOS_TOKEN], rotation=90)
# 	ax.set_yticklabels([''] + output_words.split(' ')+ [global_variables.EOS_TOKEN]);
#
# 	# Show label at every tick
# 	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# 	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
# 	plt.show()
#
# def get_encoded_batch(sentence, lang_obj, use_cuda):
# 	"""
# 	accepts only bsz = 1.
# 	input: one sentence as a string
# 	output: named tuple with vector and length
# 	"""
#
# 	sentence = sentence + ' ' + global_variables.EOS_TOKEN;
# 	tensor = lang_obj.txt2vec(sentence).unsqueeze(0)
#
# 	device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu');
#
# 	named_returntuple = namedtuple('namedtuple', ['text_vecs', 'text_lens', 'label_vecs', 'label_lens', 'use_packed'])
# 	return_tuple = named_returntuple( tensor.to(device), torch.from_numpy(np.array([tensor.shape[-1]])).to(device), None, None, False);
#
# 	return return_tuple
