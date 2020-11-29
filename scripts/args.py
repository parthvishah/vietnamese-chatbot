"""
Module for argument parcer.

Many of the arguments are from Huggingface's run_squad example:
https://github.com/huggingface/transformers/blob/7972a4019f4bc9f85fd358f42249b90f9cd27c68/examples/run_squad.py
"""
import argparse
import os

args = argparse.ArgumentParser(description='nmt translation')



args.add_argument('--experiment',
                  type=str,
                  default='testing',
                  help='name of experiment')
args.add_argument('--save_dir',
                  type=str,
                  default='results',
                  help='directory to save results')
args.add_argument('--plots_dir',
                   type=str,
                   default='plot',
                   help='directory to save results')
args.add_argument('--seed',
                  type=int,
                  default=42,
                  help='random seed')
args.add_argument('--run_log',
                  type=str,
                  default=os.path.join(os.getcwd(),'log'),
                  help='where to print run log')
args.add_argument('--access_mode',
                  type=int,
                  default=0o777,
                  help='access mode of files created')
# =============================================================================
# for dataloading
# =============================================================================

args.add_argument('--data_dir',
					type=str,
					default='data',
					help='directory storing all data')
args.add_argument('--batch_size',
					type=int,
					default=8,
					help='batch size')
# args.add_argument('--longest_label',
# 					type=int,
# 					default=1,
# 					help='longest label')
args.add_argument('--source_name',
					type=str,
					default='vi',
					help='source language name')
args.add_argument('--target_name',
					type=str,
					default='en',
					help='target language name')

# =============================================================================
# for training
# =============================================================================
args.add_argument('--learning_rate',
                  type=float,
                  default=0.25,
                  help='initial learning rate')
args.add_argument('--logging_steps',
					type=int,
					default=1e4,
					help='logs best weights every X update steps for experiment')
args.add_argument('--optimizer',
					type=str,
					default='sgd',
					help='specify what type of optimizer to use for training ("sgd","adam")')
# args.add_argument('--hidden_size',
# 					type=int,
# 					default=512,
# 					help='hidden size of RNN')
args.add_argument('--enc_emb',
					type=int,
					default=512,
					help='encoder embedding dim')
args.add_argument('--enc_hidden',
					type=int,
					default=512,
					help='encoder hidden size')
args.add_argument('--enc_layers',
					type=int,
					default=1,
					help='encoder num layers')
args.add_argument('--rnn_type',
					type=str,
					default='lstm',
					help='rnn type (lstm or gru)')
args.add_argument('--dec_emb',
					type=int,
					default=512,
					help='decoder embedding dim')
args.add_argument('--dec_hidden',
					type=int,
					default=1024,
					help='decoder hidden size')
args.add_argument('--dec_layers',
					type=int,
					default=1,
					help='decoder num layers')
# args.add_argument('--rnn_layers',
# 					type=int,
# 					default=1,
# 					help='num layers of RNN')
# args.add_argument('--gradient_clip',
# 					type=float,
# 					default=0.3,
# 					help='num layers of RNN')
args.add_argument('--epochs',
					type=int,
					default=10,
					help='number of epochs')
args.add_argument('--attn',
 					action='store_true',
 					help='with decoder attention')
args.add_argument('--wo_attn',
 					action='store_false',
 					help='without decoder attention')
args.add_argument('--beam_size',
 					type=int,
					default=3,
 					help='beam size for eval')



def check_args(parser):
    """
    make sure directories exist
    """
    assert os.path.exists(parser.data_dir), "Data directory does not exist"
    assert os.path.exists(parser.save_dir), "Save directory does not exist"
    assert os.path.exists(parser.run_log),  "Run logging directory does not exist"
