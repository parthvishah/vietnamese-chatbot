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

	# set devise to CPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
	# plots_dir = parser.plots_dir

	log.info("We will save the models in this directory: {}".format(saved_models_dir))
	# log.info("We will save the plots in this directory: {}".format(plots_dir))

	# get data dir
	main_data_path = parser.data_dir
	path_to_train_data = {'source':main_data_path+'train.tok.'+source_name, 'target':main_data_path+'train.tok.'+target_name}
	path_to_dev_data = {'source': main_data_path+'dev.tok.'+source_name, 'target':main_data_path+'dev.tok.'+target_name}

	# get language objects
	# saved_language_model_dir = os.path.join(saved_models_dir, 'lang_obj')

	# get dictionary of datasets
	#dataset_dict = {'train': nmt_dataset.LanguagePair(source_name = source_name, target_name=target_name, filepath = path_to_train_data, lang_obj_path = saved_language_model_dir, minimum_count = 1), 'dev': nmt_dataset.LanguagePair(source_name = source_name, target_name=target_name, filepath = path_to_dev_data, lang_obj_path = saved_language_model_dir, minimum_count = 1)}

	# get max sentence length by 99% percentile
	# MAX_LEN = int(dataset_dict['train'].main_df['source_len'].quantile(0.9999))
	MAX_LEN = 48
	bs = parser.batch_size
	log.info("Batch size = {}.".format(bs))

	# dataloader_dict = {'train': DataLoader(dataset_dict['train'], batch_size = batchSize, collate_fn = partial(nmt_dataset.vocab_collate_func, MAX_LEN=MAX_LEN), shuffle = True, num_workers=0), 'dev': DataLoader(dataset_dict['dev'], batch_size = batchSize, collate_fn = partial(nmt_dataset.vocab_collate_func, MAX_LEN=MAX_LEN), shuffle = True, num_workers=0)}


	# Configuration
	# source_lang_obj = dataset_dict['train'].source_lang_obj
	# target_lang_obj = dataset_dict['train'].target_lang_obj

	# source_vocab = dataset_dict['train'].source_lang_obj.n_words;
	# target_vocab = dataset_dict['train'].target_lang_obj.n_words;

	enc_emb = parser.enc_emb
	enc_hidden = parser.enc_hidden
	enc_layers = parser.enc_layers
	rnn_type = parser.rnn_type

	dec_emb = parser.dec_emb
	dec_hidden = parser.dec_hidden
	dec_layers = parser.dec_layers

	learning_rate = parser.learning_rate
	num_epochs = parser.epochs

	train, val, test, en_lang, vi_lang = dataset_helper.train_val_load(MAX_LEN, "", main_data_path)

	bs_dict = {'train':bs,'validate':1, 'train_val':1,'val_train':bs, 'test':1}
	shuffle_dict = {'train':True,'validate':False, 'train_val':False,'val_train':True, 'test':False}
	# train_used = shuffle_sorted_batches(train_sorted, bs_dict['train'])
	# train_used = train.iloc[:50]
	train_used = train
	val_used = val
	# val_used = val.iloc[:20]
	collate_fn_dict = {'train':dataset_helper.vocab_collate_func,
					   'validate':dataset_helper.vocab_collate_func_val,
					   'train_val':dataset_helper.vocab_collate_func_val,
					   'val_train':dataset_helper.vocab_collate_func,
					   'test': dataset_helper.vocab_collate_func_val}

	transformed_dataset = {'train': dataset_helper.Vietnamese(train_used),
	                       'validate': dataset_helper.Vietnamese(val_used, val = True),
	                       'train_val':dataset_helper.Vietnamese(train.iloc[:50], val = True),
	                       'val_train':dataset_helper.Vietnamese(val_used),
	                       'test':dataset_helper.Vietnamese(test, val= True)
	                                               }

	dataloader = {x: DataLoader(transformed_dataset[x], batch_size=bs_dict[x], collate_fn=collate_fn_dict[x],
	                    shuffle=shuffle_dict[x], num_workers=0) for x in ['train', 'validate', 'train_val','val_train', 'test']}

	# log.info("encoder_attention = {}, self_attention = {}".format(encoder_attention, self_attention))

	# # encoder model
	# encoder_encoderattn = nnet_models_new.EncoderRNN(input_size = source_vocab, hidden_size = hidden_size, numlayers = rnn_layers)
	#
	# # decoder model
	# decoder_encoderattn = nnet_models_new.Decoder_SelfAttn(output_size = target_vocab, hidden_size = hidden_size, encoder_attention = encoder_attention, self_attention = self_attention)
	#
	# # seq2seq model
	# nmt_encoderattn = nnet_models_new.seq2seq(encoder_encoderattn, decoder_encoderattn, lr = lr, hiddensize = hidden_size, numlayers = hidden_size, target_lang=dataset_dict['train'].target_lang_obj, longest_label = longest_label, clip = gradient_clip, device = device)

	encoder_w_att = nnet_models.EncoderRNN(vi_lang.n_words, enc_emb, enc_hidden, n_layers=enc_layers, rnn_type=rnn_type).to(device)
	decoder_w_att = nnet_models.AttentionDecoderRNN(en_lang.n_words, dec_emb, dec_hidden, n_layers=dec_layers, attention=True).to(device)

	encoder_optimizer = optim.SGD(encoder_w_att.parameters(), lr=learning_rate, nesterov=True, momentum = 0.99)
	enc_scheduler = ReduceLROnPlateau(encoder_optimizer, min_lr=1e-4, factor = 0.5, patience=0)
	decoder_optimizer = optim.SGD(decoder_w_att.parameters(), lr=learning_rate,nesterov=True, momentum = 0.99)
	dec_scheduler = ReduceLROnPlateau(decoder_optimizer, min_lr=1e-4, factor = 0.5, patience=0)
	criterion = nn.NLLLoss()

	log.info("Seq2Seq Model with the following parameters: batch_size = {}, learning_rate = {}, rnn_type = {}, enc_emb = {}, enc_hidden = {}, enc_layers = {}, dec_emb = {}, dec_hidden = {}, dec_layers = {}, num_epochs = {}, source_name = {}, target_name = {}".format(bs, learning_rate, rnn_type, enc_emb, enc_hidden, enc_layers, dec_emb, dec_hidden, dec_layers, num_epochs, source_name, target_name))

	# do we want to train again?
	train_again = False
	modelname = 'w_att'

	# if os.path.exists('lstm_wo_att_enc_1_layer.pth') and os.path.exists('lstm_wo_att_dec_1_layer.pth'):
	if os.path.exists(utils.get_full_filepath(saved_models_dir, modelname)) and (not train_again):
		log.info("Retrieving saved model from {}".format(utils.get_full_filepath(saved_models_dir, modelname)))
		encoder_w_att.load_state_dict(torch.load(saved_models_dir+"lstm_w_att_enc.pth"))
		decoder_w_att.load_state_dict(torch.load(saved_models_dir+"lstm_w_att_dec.pth"))
	else:
		log.info("Check if this path exists: {}".format(utils.get_full_filepath(saved_models_dir, modelname)))
		log.info("It does not exist! Starting to train...")
		train_utilities.train_model(encoder_optimizer, decoder_optimizer, encoder_w_att, decoder_w_att, criterion, "attention", dataloader, en_lang, vi_lang, saved_models_dir, num_epochs = num_epochs, rm = 0.95, enc_scheduler = enc_scheduler, dec_scheduler = dec_scheduler)
		log.info("Total time is: {} min : {} s".format((time.time()-start)//60, (time.time()-start)%60))
		log.info("We will save the models in this directory: {}".format(saved_models_dir))


	# generate translations
	# encoder_w_att.load_state_dict(torch.load(saved_models_dir+'lstm_w_att_enc.pth'))
	# decoder_w_att.load_state_dict(torch.load(saved_models_dir+'lstm_w_att_dec.pth'))
	#
	# log.info("Generating translations (replace_unk = False)")
	#
	# bleu_3_no_unk, att_score_no_unk_w, pred_no_unk_w, src_no_unk_w = validation_beam_search(encoder_w_att, decoder_w_att,dataloader['validate'],en_lang,\
    #                                                                   vi_lang, 'attention',3,verbose=False)
	# use_cuda = True
	# # log.info("{}".format(utils.get_translation(nmt_rnn, 'On March 14 , this year , I posted this poster on Facebook .', source_lang_obj, use_cuda, source_name, target_name)))
	# # log.info("{}".format(utils.get_translation(nmt_rnn, 'I love to watch science movies on Mondays', source_lang_obj, use_cuda, source_name, target_name)))
	# # log.info("{}".format(utils.get_translation(nmt_rnn, 'I want to be the best friend that I can be', source_lang_obj, use_cuda, source_name, target_name)))
	# # log.info("{}".format(utils.get_translation(nmt_rnn, 'I love you', source_lang_obj, use_cuda, source_name, target_name)))
	# log.info(")
	#
	# log.info("Generating translations (replace_unk = True)")
	# bleu_3_unk, att_score_unk_w, pred_unk_w, src_unk_w = validation_beam_search(encoder_w_att, decoder_w_att,dataloader['validate'],en_lang,\
    #                                                                   vi_lang, 'attention',3,verbose=False,\
    #                                                                replace_unk = True)
	# log.info("Exported Binned Bleu Score Plot to {}!".format(plots_dir))

	# export plot
	# _, _, fig = utils.get_binned_bl_score(nmt_rnn, dataset_dict['dev'], plots_dir, batchSize = batchSize)

if __name__ == "__main__":
    main()
