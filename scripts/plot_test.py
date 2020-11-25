"""
Main run script to execute NMT evaluation
"""
# =============== Import Modules ==============
import os
import time
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import partial
import sys
import logging as log
from datetime import datetime as dt
import time
import random
import numpy as np

# =============== Self Defined ===============
import global_variables
import dataset_helper
import nnet_models
import train_utilities
import bleu_score
import utils
from args import args, check_args


def main():
	start = time.time()
	parser = args.parse_args()

	# run some checks on arguments
	check_args(parser)

	# format logging
	log_name = os.path.join(parser.run_log, '{}_run_log_{}.log'.format(parser.experiment,dt.now().strftime("%Y%m%d_%H%M")))

	log.basicConfig(filename=log_name, format='%(asctime)s | %(name)s -- %(message)s', level=log.INFO)
	os.chmod(log_name, parser.access_mode)

	# set device to CPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Starting experiment {} VN -> EN NMT on {}.".format(parser.experiment,device))
	log.info("Starting experiment {} VN -> EN NMT on {}.".format(parser.experiment,device))

	# set seed for replication
	random.seed(parser.seed)
	np.random.seed(parser.seed)
	torch.manual_seed(parser.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(parser.seed)
	log.info("For reproducibility, the seed is set to {}.".format(parser.seed))

	# set file paths
	source_name = parser.source_name
	target_name = parser.target_name

	# get saved models dir
	base_saved_models_dir = parser.save_dir
	saved_models_dir = os.path.join(base_saved_models_dir, source_name+'2'+target_name)
	plots_dir = parser.plots_dir

	log.info("We will save the models in this directory: {}".format(saved_models_dir))
	log.info("We will save the plots in this directory: {}".format(plots_dir))

	# get data dir
	main_data_path = parser.data_dir
	path_to_train_data = {'source':main_data_path+'train.tok.'+source_name, 'target':main_data_path+'train.tok.'+target_name}
	path_to_dev_data = {'source': main_data_path+'dev.tok.'+source_name, 'target':main_data_path+'dev.tok.'+target_name}
	path_to_test_data = {'source': main_data_path+'test.tok.'+source_name, 'target':main_data_path+'test.tok.'+target_name}

	# Configuration
	bs = parser.batch_size
	log.info("Batch size = {}.".format(bs))

	enc_emb = parser.enc_emb
	enc_hidden = parser.enc_hidden
	enc_layers = parser.enc_layers
	rnn_type = parser.rnn_type

	dec_emb = parser.dec_emb
	dec_hidden = parser.dec_hidden
	dec_layers = parser.dec_layers

	learning_rate = parser.learning_rate
	num_epochs = parser.epochs
	attn_flag = parser.attn
	log.info("The attention flag is set to {}.".format(attn_flag))
	beam_size = parser.beam_size
	log.info("We evaluate using beam size of {}.".format(beam_size))

	train, val, test, en_lang, vi_lang = dataset_helper.train_val_load("", main_data_path)

	# get vocab sizes
	log.info('English has vocab size of: {} words.'.format(en_lang.n_words))
	log.info('Vietnamese has vocab size of: {} words.'.format(vi_lang.n_words))

	# get max sentence length by 95% percentile
	MAX_LEN = int(train['en_len'].quantile(0.95))
	log.info('We will have a max sentence length of {} (95 percentile).'.format(MAX_LEN))

	# set data loaders
	bs_dict = {'train':bs, 'validate':1, 'test':1}
	shuffle_dict = {'train':True, 'validate':False, 'test':False}

	train_used = train
	val_used = val

	collate_fn_dict = {'train':partial(dataset_helper.vocab_collate_func, MAX_LEN = MAX_LEN), 'validate':dataset_helper.vocab_collate_func_val, 'test': dataset_helper.vocab_collate_func_val}

	transformed_dataset = {'train': dataset_helper.Vietnamese(train_used), 'validate': dataset_helper.Vietnamese(val_used, val = True), 'test':dataset_helper.Vietnamese(test, val= True)}

	dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs_dict[x], collate_fn=collate_fn_dict[x], shuffle=shuffle_dict[x], num_workers=0) for x in ['train', 'validate', 'test']}

	# instantiate encoder/decoder
	encoder_w_att = nnet_models.EncoderRNN(input_size = vi_lang.n_words, embed_dim = enc_emb, hidden_size = enc_hidden, n_layers=enc_layers, rnn_type=rnn_type).to(device)
	decoder_w_att = nnet_models.AttentionDecoderRNN(output_size = en_lang.n_words, embed_dim = dec_emb, hidden_size = dec_hidden, n_layers = dec_layers, attention = attn_flag).to(device)

	# instantiate optimizer
	if parser.optimizer == 'sgd':
		encoder_optimizer = optim.SGD(encoder_w_att.parameters(), lr = learning_rate, nesterov = True, momentum = 0.99)
		decoder_optimizer = optim.SGD(decoder_w_att.parameters(), lr = learning_rate,nesterov = True, momentum = 0.99)
	elif parser.optimizer == 'adam':
		encoder_optimizer = optim.Adam(encoder_w_att.parameters(), lr = 5e-3)
		decoder_optimizer = optim.Adam(decoder_wo_att.parameters(), lr = 5e-3)
	else:
		raise ValueError('Invalid optimizer!')


	# instantiate scheduler
	enc_scheduler = ReduceLROnPlateau(encoder_optimizer, min_lr=1e-4, factor = 0.5, patience=0)
	dec_scheduler = ReduceLROnPlateau(decoder_optimizer, min_lr=1e-4, factor = 0.5, patience=0)
	criterion = nn.NLLLoss(ignore_index = global_variables.PAD_IDX)

	log.info("Seq2Seq Model with the following parameters: batch_size = {}, learning_rate = {}, rnn_type = {}, enc_emb = {}, enc_hidden = {}, enc_layers = {}, dec_emb = {}, dec_hidden = {}, dec_layers = {}, num_epochs = {}, source_name = {}, target_name = {}".format(bs, learning_rate, rnn_type, enc_emb, enc_hidden, enc_layers, dec_emb, dec_hidden, dec_layers, num_epochs, source_name, target_name))

	# do we want to train again?
	train_again = False
	encoder_save = '{}_att_{}bs_{}_enc_{}_layer'.format(rnn_type, bs, parser.optimizer, enc_layers)
	decoder_save = '{}_att_{}bs_{}_dec_{}_layer'.format(rnn_type, bs, parser.optimizer, dec_layers)

	print(utils.get_full_filepath(saved_models_dir, encoder_save))
 	print(utils.get_full_filepath(saved_models_dir, encoder_save))
	log.info(utils.get_full_filepath(saved_models_dir, encoder_save))
 	log.info(utils.get_full_filepath(saved_models_dir, encoder_save))
	encoder_w_att.load_state_dict(torch.load(utils.get_full_filepath(saved_models_dir, encoder_save)))
	decoder_w_att.load_state_dict(torch.load(utils.get_full_filepath(saved_models_dir, decoder_save)))


	# BLEU with beam size
	bleu_no_unk, att_score_wo, pred_wo, src_wo = train_utilities.validation_beam_search(encoder_w_att, decoder_w_att, dataloader['validate'], en_lang, vi_lang, 'attention', beam_size, verbose = False)

	log.info("Bleu-{} Score (No UNK): {}".format(beam_size, bleu_no_unk))
	print("Bleu-{} Score (No UNK): {}".format(beam_size, bleu_no_unk))

	bleu_unk, att_score_wo, pred_wo, src_wo = train_utilities.validation_beam_search(encoder_w_att, decoder_w_att,dataloader['validate'], en_lang, vi_lang, 'attention', beam_size, verbose = False, replace_unk = True)

	log.info("Bleu-{} Score (UNK): {}".format(beam_size, bleu_unk))
	print("Bleu-{} Score (UNK): {}".format(beam_size, bleu_unk))

	# generate 5 random predictions
	indexes = range(len(pred_wo))
	for i in np.random.choice(indexes, 5):
		print('Source: {} \nPrediction: {}\n---'.format(src_wo[i], pred_wo[i]))
		log.info('Source: {} \nPrediction: {}\n---'.format(src_wo[i], pred_wo[i]))

	log.info("Exported Binned Bleu Score Plot to {}!".format(plots_dir))
	_, _, fig = utils.get_binned_bl_score(ncoder_w_att, decoder_w_att, transformed_dataset['validate'], attn_flag, beam_size, plots_dir)


if __name__ == "__main__":
	main()
