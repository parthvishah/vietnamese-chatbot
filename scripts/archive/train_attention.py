"""
Main run script to execute NMT evaluation
"""
# =============== Import Modules ==============
import os
import time
import torch
from torch.utils.data import DataLoader
from functools import partial
import sys
import logging as log
from datetime import datetime as dt
import time
import random
import numpy as np

# =============== Self Defined ===============
import global_variables
import nmt_dataset
import nnet_models_new
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
	plots_dir = parser.plots_dir

	log.info("We will save the models in this directory: {}".format(saved_models_dir))
	log.info("We will save the plots in this directory: {}".format(plots_dir))

	# get data dir
	main_data_path = parser.data_dir
	path_to_train_data = {'source':main_data_path+'train.'+source_name, 'target':main_data_path+'train.'+target_name}
	path_to_dev_data = {'source': main_data_path+'dev.'+source_name, 'target':main_data_path+'dev.'+target_name}
	# get language objects
	saved_language_model_dir = os.path.join(saved_models_dir, 'lang_obj')

	# get dictionary of datasets
	dataset_dict = {'train': nmt_dataset.LanguagePair(source_name = source_name, target_name=target_name, filepath = path_to_train_data, lang_obj_path = saved_language_model_dir, minimum_count = 1), 'dev': nmt_dataset.LanguagePair(source_name = source_name, target_name=target_name, filepath = path_to_dev_data, lang_obj_path = saved_language_model_dir, minimum_count = 1)}

	# get max sentence length by 99% percentile
	MAX_LEN = int(dataset_dict['train'].main_df['source_len'].quantile(0.9999))
	batchSize = parser.batch_size
	log.info("Batch size = {}.".format(batchSize))

	dataloader_dict = {'train': DataLoader(dataset_dict['train'], batch_size = batchSize, collate_fn = partial(nmt_dataset.vocab_collate_func, MAX_LEN=MAX_LEN), shuffle = True, num_workers=0), 'dev': DataLoader(dataset_dict['dev'], batch_size = batchSize, collate_fn = partial(nmt_dataset.vocab_collate_func, MAX_LEN=MAX_LEN), shuffle = True, num_workers=0)}

	# Configuration
	source_lang_obj = dataset_dict['train'].source_lang_obj
	target_lang_obj = dataset_dict['train'].target_lang_obj

	source_vocab = dataset_dict['train'].source_lang_obj.n_words;
	target_vocab = dataset_dict['train'].target_lang_obj.n_words;
	hidden_size = parser.hidden_size
	rnn_layers = parser.rnn_layers
	lr = parser.learning_rate
	longest_label = parser.longest_label
	gradient_clip = parser.gradient_clip
	num_epochs = parser.epochs
	encoder_attention = parser.encoder_attention
	self_attention = parser.self_attention

	log.info("encoder_attention = {}, self_attention = {}".format(encoder_attention, self_attention))

	# encoder model
	encoder_encoderattn = nnet_models_new.EncoderRNN(input_size = source_vocab, hidden_size = hidden_size, numlayers = rnn_layers)

	# decoder model
	decoder_encoderattn = nnet_models_new.Decoder_SelfAttn(output_size = target_vocab, hidden_size = hidden_size, encoder_attention = encoder_attention, self_attention = self_attention)

	# seq2seq model
	nmt_encoderattn = nnet_models_new.seq2seq(encoder_encoderattn, decoder_encoderattn, lr = lr, hiddensize = hidden_size, numlayers = hidden_size, target_lang=dataset_dict['train'].target_lang_obj, longest_label = longest_label, clip = gradient_clip, device = device)

	log.info("Seq2Seq Model with the following parameters: encoder_attention = {}, self_attention = {}, batch_size = {}, learning_rate = {}, hidden_size = {}, rnn_layers = {}, lr = {}, longest_label = {}, gradient_clip = {}, num_epochs = {}, source_name = {}, target_name = {}".format(encoder_attention, self_attention, batchSize, lr, hidden_size, rnn_layers, lr, longest_label, gradient_clip, num_epochs, source_name, target_name))

	# do we want to train again?
	train_again = False
	modelname = 'encoderattn'

	# check if there is a saved model and if we want to train again
	if os.path.exists(utils.get_full_filepath(saved_models_dir, modelname)) and (not train_again):
		log.info("Retrieving saved model from {}".format(utils.get_full_filepath(saved_models_dir, modelname)))
		nmt_encoderattn = torch.load(utils.get_full_filepath(saved_models_dir, modelname))
	# train model again
	else:
		log.info("Check if this path exists: {}".format(utils.get_full_filepath(saved_models_dir, modelname)))
		log.info("It does not exist! Starting to train...")
		utils.train_model(dataloader_dict, nmt_encoderattn, num_epochs = num_epochs, saved_model_path = saved_models_dir, enc_type = 'encoderattn_test')
	log.info("Total time is: {} min : {} s".format((time.time()-start)//60, (time.time()-start)%60))
	log.info("We will save the models in this directory: {}".format(saved_models_dir))

	# generate translations
	use_cuda = True
	utils.get_translation(nmt_encoderattn, 'I love to watch science movies on Mondays', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'I want to be the best friend that I can be', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'I love you', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'I love football, I like to watch it with my friends. It is always a great time.', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'I do not know what I would do without pizza, it is very tasty to eat. If I could have any food in the world it would probably be pizza.', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'Trump is the worst president in all of history. He can be a real racist and say very nasty things to people of color.', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'Thank you very much.', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'Think about your own choices.', source_lang_obj, use_cuda, source_name, target_name)
	utils.get_translation(nmt_encoderattn, 'I recently did a survey with over 2,000 Americans , and the average number of choices that the typical American reports making is about 70 in a typical day .', source_lang_obj, use_cuda, source_name, target_name)
	# export plot
	_, _, fig = utils.get_binned_bl_score(nmt_encoderattn, dataset_dict['dev'], plots_dir, batchSize = batchSize)

if __name__ == "__main__":
    main()
