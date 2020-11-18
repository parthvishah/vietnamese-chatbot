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
import global_variables
from args import args, check_args


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

        print('Epoch: [{}/{}]'.format(epoch, num_epochs))

        for i, data in enumerate(dataloader['train']):
            _, curr_loss = nmt.train_step(data)
            running_loss += curr_loss

        epoch_loss = running_loss / len(dataloader['train'])

        print("epoch {} loss = {}, time = {}".format(epoch, epoch_loss, time.time() - start))

        sys.stdout.flush()

        if epoch%val_every == 0:
            val_bleu_score = nmt.get_bleu_score(dataloader['dev'])
            print('validation bleu: ', val_bleu_score)
            sys.stdout.flush()

            nmt.scheduler_step(val_bleu_score)

            if val_bleu_score > best_bleu:
                best_bleu = val_bleu_score
                save_models(nmt, saved_model_path, enc_type)
            log.info(f"epoch {epoch} | loss {epoch_loss} | time = {time.time() - start} | validation bleu = {val_bleu_score}")

        print('='*50)

    print("Training completed. Best BLEU is {}".format(best_bleu))


def get_binned_bl_score(nmt_model, val_dataset, batchSize, nmt_dataset, save_plots):
    len_threshold = np.arange(0, 31, 5)
    bin_bl_score = np.zeros(len(len_threshold))
    
    for i in range(1, len(len_threshold)):
        log.info('entered for loop')
        min_len = len_threshold[i-1]
#         	min_len = 0
        max_len = len_threshold[i]
        temp_dataset = copy.deepcopy(val_dataset)
        temp_dataset.main_df = temp_dataset.main_df[(temp_dataset.main_df['source_len'] > min_len) & (temp_dataset.main_df['source_len'] <= max_len)]
        temp_loader = DataLoader(temp_dataset, batch_size = batchSize, 
                        collate_fn = partial(nmt_dataset.vocab_collate_func, MAX_LEN=100),
                        shuffle = True, num_workers=0)
        log.info('get_bleu_score')
        bin_bl_score[i] = nmt_model.get_bleu_score(temp_loader)
    
    
    len_threshold = len_threshold[1:]
    bin_bl_score = bin_bl_score[1:]
    
    plt.plot(len_threshold, bin_bl_score, 'x-')
    plt.ylim(0, np.max(bin_bl_score)+1)
    plt.xlabel('len')
    plt.ylabel('bl score')
    script_dir = save_plots
    saved_models_dir = os.path.join(script_dir, 'binnedplot')
    if not os.path.isdir(saved_models_dir):
        os.makedirs(saved_models_dir)
    plt.savefig(results_dir)
    
    return len_threshold, bin_bl_score

def showAttention(input_sentence, output_words, attentions, save_plots):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone', aspect='auto')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       [global_variables.EOS_TOKEN], rotation=90)
    ax.set_yticklabels([''] + output_words.split(' ')+
                       [global_variables.EOS_TOKEN])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    script_dir = save_plots
    saved_models_dir = os.path.join(script_dir, 'attention')
    if not os.path.isdir(saved_models_dir):
        os.makedirs(saved_models_dir)
    plt.savefig(results_dir)

    
    
def get_encoded_batch(sentence, lang_obj, use_cuda):
    
    sentence = sentence + ' ' + global_variables.EOS_TOKEN
    tensor = source_lang_obj.txt2vec(sentence).unsqueeze(0)
    
    device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
    
    named_returntuple = namedtuple('namedtuple', ['text_vecs', 'text_lens', 'label_vecs', 'label_lens', 'use_packed'])
    return_tuple = named_returntuple( tensor.to(device), 
                                     torch.from_numpy(np.array([tensor.shape[-1]])).to(device),
                                     None,
                                     None,
                                     False )

    return return_tuple

def get_translation(nmt_model, sentence, lang_obj, use_cuda, save_plots):
    print('souce: ', sentence)
    batch = get_encoded_batch(sentence, lang_obj, use_cuda)
    prediction, attn_scores_list = nmt_model.eval_step(batch, return_attn = True)
    prediction = prediction[0]
    print('prediction: ', prediction)
    
    if attn_scores_list[0] is not None:
        if attn_scores_list[0][0] is not None:
            attn_matrix = [x[0].data.cpu().numpy() for x in attn_scores_list]
            attn_matrix = np.stack(attn_matrix)[:,:, 0]
            showAttention(sentence, prediction, attn_matrix, save_plots)
