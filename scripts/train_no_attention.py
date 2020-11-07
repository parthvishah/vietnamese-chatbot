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
	log_name = os.path.join(parser.run_log,
							'{}_nmt_run_log_{}.log'.format(parser.experiment,dt.now().strftime("%Y%m%d_%H%M")))

	log.basicConfig(filename=log_name,
					format='%(asctime)s | %(name)s -- %(message)s',
					level=log.INFO)
	os.chmod(log_name, parser.access_mode)

	# set seed for replication
	random.seed(parser.seed)
	np.random.seed(parser.seed)
	torch.manual_seed(parser.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(parser.seed)

	# set devise to CPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	log.info("Device is {}".format(device))

	log.info("Starting experiment {} VN -> EN NMT on {}".format(parser.experiment,device))


	# set file paths
	source_name = parser.source_name
	target_name = parser.target_name

	# get saved models dir
	base_saved_models_dir = parser.save_dir
	saved_models_dir = os.path.join(base_saved_models_dir, source_name+'2'+target_name)

	# get data dir
	main_data_path = parser.data_dir
	path_to_train_data = {'source':main_data_path+'train.'+source_name,
							'target':main_data_path+'train.'+target_name}
	path_to_dev_data = {'source': main_data_path+'dev.'+source_name,
							'target':main_data_path+'dev.'+target_name}
	# get language objects
	saved_language_model_dir = os.path.join(saved_models_dir, 'lang_obj')

	# get dictionary of datasets
	dataset_dict = {'train': nmt_dataset.LanguagePair(source_name = source_name,
													target_name=target_name,
													filepath = path_to_train_data,
													lang_obj_path = saved_language_model_dir,
													minimum_count = 1),
					'dev': nmt_dataset.LanguagePair(source_name = source_name,
													target_name=target_name,
													filepath = path_to_dev_data,
													lang_obj_path = saved_language_model_dir,
													minimum_count = 1)}

	# get max sentence length bby 99% percentile
	MAX_LEN = int(dataset_dict['train'].main_df['source_len'].quantile(0.9999))
	batchSize = parser.batch_size

	dataloader_dict = {'train': DataLoader(dataset_dict['train'],
											batch_size = batchSize,
											collate_fn = partial(nmt_dataset.vocab_collate_func, MAX_LEN=MAX_LEN),
											shuffle = True, num_workers=0),
	                    'dev': DataLoader(dataset_dict['dev'],
											batch_size = batchSize,
											collate_fn = partial(nmt_dataset.vocab_collate_func, MAX_LEN=MAX_LEN),
											shuffle = True, num_workers=0)}

	# Configuration
	source_lang_obj = dataset_dict['train'].source_lang_obj
	target_lang_obj = dataset_dict['train'].target_lang_obj

	source_vocab = dataset_dict['train'].source_lang_obj.n_words;
	target_vocab = dataset_dict['train'].target_lang_obj.n_words;
	hidden_size = parser.hidden_size
	rnn_layers = parser.rnn_layers
	lr = 0.25
	longest_label = parser.longest_label
	gradient_clip = parser.gradient_clip
	use_cuda = True

	num_epochs = parser.epochs

	# encoder model
	encoder_rnn = nnet_models_new.EncoderRNN(input_size = source_vocab,
											hidden_size = hidden_size,
											numlayers = rnn_layers)
	# decoder model
	decoder_rnn = nnet_models_new.DecoderRNN(output_size = target_vocab,
												hidden_size = hidden_size,
												numlayers = rnn_layers)

	# seq2seq model
	nmt_rnn = nnet_models_new.seq2seq(encoder_rnn,
									decoder_rnn,
									lr = lr,
									use_cuda = use_cuda,
									hiddensize = hidden_size,
									numlayers = hidden_size,
									target_lang=dataset_dict['train'].target_lang_obj,
									longest_label = longest_label,
									clip = gradient_clip,
									device = device)

	train_again = False
	if os.path.exists(utils.get_full_filepath(saved_models_dir, 'rnn')) and (not train_again):
		nmt_rnn = torch.load(utils.get_full_filepath(saved_models_dir, 'rnn'), map_location=global_variables.device)
	else:
		utils.train_model(dataloader_dict,
							nmt_rnn,num_epochs = num_epochs,
							saved_model_path = saved_models_dir,
							enc_type = 'rnn_test')
	log.info("Total time is: {} min : {} s".format((time.time()-start)//60, (time.time()-start)%60))

if __name__ == "__main__":
    main()
