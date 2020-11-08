import os
import time
import torch
from torch.utils.data import DataLoader
from functools import partial
import sys
import logging as log
from datetime import datetime as dt
import time

def get_full_filepath(path, enc_type):
	'''
	get the full checkpoint file path
	'''
	filename = 'nmt_enc_'+enc_type+'_dec_rnn.pth'
	return os.path.join(path, filename)

def save_models(nmt_model, path, enc_type):
	'''
	save the model
	'''
	if not os.path.exists(path):
		os.makedirs(path)
	filename = 'nmt_enc_'+enc_type+'_dec_rnn.pth'
	torch.save(nmt_model, os.path.join(path, filename))

def train_model(dataloader, nmt, num_epochs=50, val_every=1, saved_model_path = '.', enc_type ='rnn'):
	'''
	nmt training loop
	'''
	best_bleu = -1;
	for epoch in range(num_epochs):

		start = time.time()
		running_loss = 0

		print('Epoch: [{}/{}]'.format(epoch, num_epochs));

		for i, data in enumerate(dataloader['train']):
			_, curr_loss = nmt.train_step(data);
			running_loss += curr_loss

		epoch_loss = running_loss / len(dataloader['train'])

		print("epoch {} loss = {}, time = {}".format(epoch, epoch_loss,
														time.time() - start))

		sys.stdout.flush()

		if epoch%val_every == 0:
			val_bleu_score = nmt.get_bleu_score(dataloader['dev']);
			print('validation bleu: ', val_bleu_score)
			sys.stdout.flush()

			nmt.scheduler_step(val_bleu_score);

			if val_bleu_score > best_bleu:
				best_bleu = val_bleu_score
				save_models(nmt, saved_model_path, enc_type);
			log.info(f"epoch {epoch} | loss {epoch_loss} | time = {time.time() - start} | validation bleu = {val_bleu_score}")

		print('='*50)

	print("Training completed. Best BLEU is {}".format(best_bleu))
