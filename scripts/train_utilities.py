import torch
import time
import random
import errno
import sys
import copy
import numpy as np
import utils
import tqdm
import logging as log

import global_variables
from bleu_score import BLEU_SCORE

"""
This .py file contains the following functions:

	convert_idx_2_sent(tensor, lang_obj)
	convert_id_list_2_sent(list_idx, lang_obj)
	validation_new(encoder, decoder, val_dataloader, lang_en,lang_vi,m_type, verbose = False, replace_unk = False)
	validation_beam_search(encoder, decoder, val_dataloader,lang_en,lang_vi,m_type, beam_size, verbose = False,\
	                           device = 'cuda', replace_unk = False)
	encode_decode(encoder,decoder,data_en,data_de,src_len,tar_len,rand_num = 0.95, val = False)
	train_model(encoder_optimizer,decoder_optimizer, encoder, decoder, loss_fun,m_type, dataloader, en_lang,vi_lang,\
	                num_epochs=60, val_every = 1, train_bleu_every = 10,clip = 0.1, rm = 0.8, enc_scheduler = None,\
	               dec_scheduler = None, enc_dec_fn = encode_decode, val_fn = validation_new)
	flatten_cel_loss(input,target,nll)
"""

SOS_token = global_variables.SOS_token
EOS_token = global_variables.EOS_token
UNK_IDX = global_variables.UNK_IDX
PAD_IDX = global_variables.PAD_IDX

device = global_variables.device;


def convert_idx_2_sent(tensor, lang_obj):
	'''
	'''
	word_list = []
	for i in tensor:
		if i.item() not in set([PAD_IDX,EOS_token,SOS_token]):
			word_list.append(lang_obj.index2word[i.item()])
	return (' ').join(word_list)


def convert_id_list_2_sent(list_idx, lang_obj):
	'''
	'''
	word_list = []
	if type(list_idx) == list:
		for i in list_idx:
			if i not in set([EOS_token]):
				word_list.append(lang_obj.index2word[i])
	else:
		for i in list_idx:
			if i.item() not in set([EOS_token,SOS_token,PAD_IDX]):
				word_list.append(lang_obj.index2word[i.item()])
	return (' ').join(word_list)

def validation_new(encoder, decoder, val_dataloader, lang_en,lang_vi,m_type, verbose = False, replace_unk = False):
	encoder.eval()
	decoder.eval()
	pred_corpus = []
	true_corpus = []
	src_corpus = []
	running_loss = 0
	running_total = 0
	bl = BLEU_SCORE()
	attention_scores_for_all_val = []
	for data in val_dataloader:
		encoder_i = data[0].to(device)
		src_len = data[2].to(device)
		bs,sl = encoder_i.size()[:2]
		en_out,en_hid,en_c = encoder(encoder_i,src_len)
		max_src_len_batch = max(src_len).item()
		prev_hiddens = en_hid
		prev_cs = en_c
		decoder_input = torch.tensor([[SOS_token]]*bs).to(device)
		prev_output = torch.zeros((bs, en_out.size(-1))).to(device)
		d_out = []
		attention_scores = []
		for i in range(sl*2):
			out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder(decoder_input, prev_output, prev_hiddens, prev_cs, en_out, src_len)
			topv, topi = out_vocab.topk(1)
			d_out.append(topi.item())
			decoder_input = topi.squeeze().detach().view(-1,1)
			if m_type == 'attention':
				attention_scores.append(attention_score.unsqueeze(-1))
			if topi.item() == EOS_token:
				break
		if replace_unk:
			true_sent = convert_id_list_2_sent(data[1][0],lang_en)
			true_corpus.append(true_sent)
		else:
			true_corpus.append(data[-1])
		src_sent = convert_id_list_2_sent(data[0][0],lang_vi)
		src_corpus.append(src_sent)
		pred_sent = convert_id_list_2_sent(d_out,lang_en)
		pred_corpus.append(pred_sent)
		if m_type == 'attention':
			attention_scores = torch.cat(attention_scores, dim = -1)
			attention_scores_for_all_val.append(attention_scores)
		if verbose:
			print("True Sentence:",data[-1])
			print("Pred Sentence:", pred_sent)
			print('-*'*50)
	score = bl.corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]
	return score, attention_scores_for_all_val, pred_corpus, src_corpus



def validation_beam_search(encoder, decoder, val_dataloader,lang_en,lang_vi,m_type, beam_size, verbose = False,\
                           device = 'cuda', replace_unk = False):
    encoder.eval()
    decoder.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    pred_corpus = []
    true_corpus = []
    src_corpus = []
    running_loss = 0
    running_total = 0
    bl = BLEU_SCORE()
    j = 0
    attention_scores_for_all_val = []
    for data in val_dataloader:
#         print(j)
        encoder_i = data[0].to(device)
        src_len = data[2].to(device)

        bs,sl = encoder_i.size()[:2]
        en_out,en_hid,en_c = encoder(encoder_i,src_len)
        max_src_len_batch = max(src_len).item()
        prev_hiddens = en_hid
        prev_cs = en_c
        decoder_input = torch.tensor([[SOS_token]]*bs).to(device)
        prev_output = torch.zeros((bs, en_out.size(-1))).to(device)

        list_decoder_input = [None]*beam_size
        beam_stop_flags = [False]*beam_size
        beam_score = torch.zeros((bs,beam_size)).to(device)
        list_d_outs = [[] for _ in range(beam_size)]
        select_beam_size = beam_size
        attention_scores = [[] for _ in range(beam_size)]
        for i in range(sl+20):
            if i == 0:
                out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, \
                                                                                    prev_hiddens,prev_cs, en_out,\
                                                                                    src_len)
                bss, vocab_size = out_vocab.size()
                topv, topi = out_vocab.topk(beam_size)
                list_prev_output = [prev_output]*beam_size
                list_prev_hiddens = [prev_hiddens]*beam_size
                list_prev_cs = [prev_cs]*beam_size
                for b in range(beam_size):
                    beam_score[0][b] = topv[0][b].item()
                    list_decoder_input[b] = topi[0][b].squeeze().detach().view(-1,1)
                    list_d_outs[b].append(topi[0][b].item())
                    if m_type == 'attention':
                        attention_scores[b].append(attention_score.unsqueeze(-1))
                    if topi[0][b].item() == EOS_token:
                        beam_stop_flags[b] = True
            else:
                beam_out_vocab = [None]*beam_size
                temp_out = [None]*beam_size
                temp_hid = [None]*beam_size
                temp_c = [None]*beam_size
                temp_attention_score = [[] for _ in range(beam_size)]
                prev_d_outs = copy.deepcopy(list_d_outs)
                for b in range(beam_size):
                    if not beam_stop_flags[b]:
                        beam_out_vocab[b], temp_out[b], temp_hid[b], temp_c[b], temp_attention_score[b] =\
                            decoder(list_decoder_input[b],list_prev_output[b],list_prev_hiddens[b],list_prev_cs[b],\
                                    en_out,src_len)
                        beam_out_vocab[b] = beam_out_vocab[b] + beam_score[0][b]
                    if beam_stop_flags[b]:
                        beam_out_vocab[b] = torch.zeros(bss,vocab_size).fill_(float('-inf')).to(device)
                beam_out_vocab = torch.cat(beam_out_vocab,dim = 1)

                topv, topi = beam_out_vocab.topk(beam_size)
                id_for_hid = topi//vocab_size
                topi_idx = topi%vocab_size
                for b in range(beam_size):
                    if not beam_stop_flags[b]:
                        beam_score[0][b] = topv[0][b].item()
                        list_decoder_input[b] = topi_idx[0][b].squeeze().detach().view(-1,1)
                        list_d_outs[b] = copy.deepcopy(prev_d_outs[id_for_hid[0][b]])
                        list_d_outs[b].append(topi_idx[0][b].item())
                        if m_type == 'attention':
                            attention_scores[b].append(temp_attention_score[b].unsqueeze(-1))
                        if topi_idx[0][b].item() == EOS_token:
                            beam_stop_flags[b] = True
                        else:
                            list_prev_output[b] = temp_out[id_for_hid[0][b]]
                            list_prev_hiddens[b] = temp_hid[id_for_hid[0][b]]
                            list_prev_cs[b] = temp_c[id_for_hid[0][b]]
                if all(beam_stop_flags):
                    break

        id_max_score = torch.argmax(beam_score)
        d_out = list_d_outs[id_max_score]
        if m_type == 'attention':

            att_score = attention_scores[id_max_score]
            att_score = torch.cat(att_score, dim = -1)
            attention_scores_for_all_val.append(att_score)
        if replace_unk:
            true_sent = convert_id_list_2_sent(data[1][0],lang_en)
            true_corpus.append(true_sent)
        else:
            true_corpus.append(data[-1])
        pred_sent = convert_id_list_2_sent(d_out,lang_en)
        pred_corpus.append(pred_sent)
        src_sent = convert_id_list_2_sent(data[0][0], lang_vi)
        src_corpus.append(src_sent)
        if verbose:
            print("True Sentence:",data[-1])
            print("Pred Sentence:", pred_sent)
            print('-*'*50)

    score = bl.corpus_bleu(pred_corpus,[true_corpus],lowercase=True)[0]
    return score, attention_scores_for_all_val, pred_corpus, src_corpus

def encode_decode(encoder, decoder, data_en, data_de, src_len, tar_len, rand_num = 0.95, val = False):
	if not val:
		use_teacher_forcing = True if random.random() < rand_num else False

		bss = data_en.size(0)
		en_out,en_hid,en_c = encoder(data_en, src_len)
		max_src_len_batch = max(src_len).item()
		max_tar_len_batch = max(tar_len).item()
		prev_hiddens = en_hid
		prev_cs = en_c
		decoder_input = torch.tensor([[SOS_token]]*bss).to(device)
		prev_output = torch.zeros((bss, en_out.size(-1))).to(device)
		if use_teacher_forcing:
			d_out = []
			for i in range(max_tar_len_batch):
				out_vocab, prev_output, prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, prev_hiddens,prev_cs, en_out, src_len)
				d_out.append(out_vocab.unsqueeze(-1))
				decoder_input = data_de[:,i].view(-1,1)
			d_out = torch.cat(d_out,dim=-1)
		else:
			d_out = []
			for i in range(max_tar_len_batch):
				out_vocab, prev_output, prev_hiddens, prev_cs, attention_score = decoder(decoder_input, prev_output, prev_hiddens, prev_cs, en_out, src_len)
				d_out.append(out_vocab.unsqueeze(-1))
				topv, topi = out_vocab.topk(1)
				decoder_input = topi.squeeze().detach().view(-1,1)
			d_out = torch.cat(d_out,dim=-1)
		return d_out
	else:
		encoder.eval()
		decoder.eval()
		bss = data_en.size(0)
		en_out,en_hid,en_c = encoder(data_en, src_len)
		max_src_len_batch = max(src_len).item()
		max_tar_len_batch = max(tar_len).item()
		prev_hiddens = en_hid
		prev_cs = en_c
		decoder_input = torch.tensor([[SOS_token]]*bss).to(device)
		prev_output = torch.zeros((bss, en_out.size(-1))).to(device)
		d_out = []
		for i in range(max_tar_len_batch):
			out_vocab, prev_output, prev_hiddens, prev_cs, attention_score = decoder(decoder_input,prev_output, prev_hiddens, prev_cs, en_out, src_len)
			d_out.append(out_vocab.unsqueeze(-1))
			topv, topi = out_vocab.topk(1)
			decoder_input = topi.squeeze().detach().view(-1,1)
		d_out = torch.cat(d_out,dim=-1)
		return d_out


def train_model(encoder_optimizer, decoder_optimizer, encoder, decoder, loss_fun, m_type, dataloader, en_lang, vi_lang, save_path, encoder_save, decoder_save, num_epochs=60, val_every = 1, train_bleu_every = 10,clip = 0.1, rm = 0.8, enc_scheduler = None, dec_scheduler = None, enc_dec_fn = encode_decode, val_fn = validation_new):
	'''
	'''
	best_score = 0
	best_bleu = 0
	loss_hist = {'train': [], 'validate': []}
	bleu_hist = {'train': [], 'validate': []}
	best_encoder_wts = None
	best_decoder_wts = None
	phases = ['train','validate']
	for epoch in range(num_epochs):
		print('Epoch: [{}/{}]'.format(epoch, num_epochs));
		log.info('Epoch: [{}/{}]'.format(epoch, num_epochs))
		for ex, phase in enumerate(phases):
			start = time.time()
			total = 0
			top1_correct = 0
			running_loss = 0
			running_total = 0
			if phase == 'train':
				encoder.train()
				decoder.train()
			else:
				encoder.eval()
				decoder.eval()

			for i, data in enumerate(dataloader[phase]):

				encoder_optimizer.zero_grad()
				decoder_optimizer.zero_grad()

				encoder_i = data[0].to(device)
				decoder_i = data[1].to(device)
				src_len = data[2].to(device)
				tar_len = data[3].to(device)
				if phase == 'validate':
					out = enc_dec_fn(encoder, decoder, encoder_i, decoder_i, src_len, tar_len, rand_num=rm, val = True)
				else:
					out = enc_dec_fn(encoder, decoder, encoder_i, decoder_i, src_len, tar_len, rand_num=rm, val = False)
				N = decoder_i.size(0)
				loss = loss_fun(out.float(), decoder_i.long())
				running_loss += loss.item() * N

				total += N
				if phase == 'train':
					loss.backward()
					torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
					torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
					encoder_optimizer.step()
					decoder_optimizer.step()

			epoch_loss = running_loss / total
			loss_hist[phase].append(epoch_loss)
			print("epoch {} {} loss = {}, time = {}".format(epoch, phase, epoch_loss, time.time() - start))
			log.info("epoch {} {} loss = {}, time = {}".format(epoch, phase, epoch_loss, time.time() - start))


		if (enc_scheduler is not None) and (dec_scheduler is not None):
			enc_scheduler.step(loss_hist['train'][-1])
			dec_scheduler.step(loss_hist['train'][-1])

		if epoch%val_every == 0:
			val_bleu_score, _, _ , _ = val_fn(encoder, decoder, dataloader['validate'], en_lang, vi_lang, m_type, verbose=False, replace_unk=True)
			bleu_hist['validate'].append(val_bleu_score)
			print("validation BLEU = {}".format(val_bleu_score))
			log.info("validation BLEU = {}".format(val_bleu_score))
			if val_bleu_score > best_bleu:
				best_bleu = val_bleu_score
				best_encoder_wts = encoder.state_dict()
				best_decoder_wts = decoder.state_dict()
				# save best model
				utils.save_models(best_encoder_wts, save_path, encoder_save)
				utils.save_models(best_decoder_wts, save_path, decoder_save)
		print('='*50)
	encoder.load_state_dict(best_encoder_wts)
	decoder.load_state_dict(best_decoder_wts)
	print("Training completed. Best BLEU is {}".format(best_bleu))
	return encoder, decoder, loss_hist, bleu_hist

def flatten_cel_loss(input,target,nll):
	input = input.transpose(1,2)
	bs, sl = input.size()[:2]
	return nll(input.contiguous().view(bs*sl,-1),target.contiguous().view(-1))
