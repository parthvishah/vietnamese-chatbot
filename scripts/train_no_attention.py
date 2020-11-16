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

	log.info("We will save the models in this directory: {}".format(saved_models_dir))

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

	# encoder model
	encoder_rnn = nnet_models_new.EncoderRNN(input_size = source_vocab, hidden_size = hidden_size, numlayers = rnn_layers)
	# decoder model
	decoder_rnn = nnet_models_new.DecoderRNN(output_size = target_vocab, hidden_size = hidden_size, numlayers = rnn_layers)

	# seq2seq model
	nmt_rnn = nnet_models_new.seq2seq(encoder_rnn, decoder_rnn, lr = lr, hiddensize = hidden_size, numlayers = hidden_size, target_lang=dataset_dict['train'].target_lang_obj, longest_label = longest_label, clip = gradient_clip, device = device)

	log.info("Seq2Seq Model with the following parameters: batch_size = {}, learning_rate = {}, hidden_size = {}, rnn_layers = {}, lr = {}, longest_label = {}, gradient_clip = {}, num_epochs = {}, source_name = {}, target_name = {}".format(batchSize, lr, hidden_size, rnn_layers, lr, longest_label, gradient_clip, num_epochs, source_name, target_name))

	# do we want to train again?
	train_again = False

	# check if there is a saved model and if we want to train again
	if os.path.exists(utils.get_full_filepath(saved_models_dir, 'rnn')) and (not train_again):
		log.info("Retrieving saved model from {}".format(utils.get_full_filepath(saved_models_dir, 'rnn')))
		nmt_rnn = torch.load(utils.get_full_filepath(saved_models_dir, 'rnn'), map_location=global_variables.device)
	# train model again
	else:
		log.info("Check if this path exists: {}".format(utils.get_full_filepath(saved_models_dir, 'rnn')))
		log.info("It does not exist! Starting to train...")
		utils.train_model(dataloader_dict, nmt_rnn,num_epochs = num_epochs, saved_model_path = saved_models_dir, enc_type = 'rnn_test')
	log.info("Total time is: {} min : {} s".format((time.time()-start)//60, (time.time()-start)%60))
	
	rnn_len_threshold, rnn_bin_bl = utils.get_binned_bl_score(nmt_rnn, dataset_dict['dev'])
	#for i in range(10):
	utils.get_translation(nmt_rnn, 'are hello ? how you', source_lang_obj, use_cuda)
	
if __name__ == "__main__":
    main()
