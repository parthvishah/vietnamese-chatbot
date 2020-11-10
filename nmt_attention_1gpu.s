#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,k80_4
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ltk224@nyu.edu
#SBATCH --job-name="seq2seq_attn_train"
#SBATCH --output=/scratch/ltk224/nlp_project/outputs/%j.out

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=ltk224
source activate nmt_env

# Set project working directory
PROJECT=/scratch/${NETID}/nlp_project

# Set arguments
STUDY_NAME=nmt_attn_64b_0.25lr_1gpu #nmt_attn_batchsize_learningrate_gpus
SAVE_DIR=${PROJECT}/saved_models
DATA_DIR=${PROJECT}/vietnamese-chatbot/data/interim/iwslt15-en-vn/
BATCH_SIZE=64
LR=0.25
SEED=42
SOURCE_NAME='en'
TARGET_NAME='vi'
HIDDEN_SIZE=512
RNN_LAYERS=1
LONGEST_LABEL=1
GRADIENT_CLIP=0.3
EPOCHS=10
ENCODER_ATTN=TRUE
SELF_ATTN=TRUE

cd ${PROJECT}
python ./vietnamese-chatbot/scripts/train_attention.py \
	--experiment ${STUDY_NAME} \
	--save_dir ${SAVE_DIR} \
	--data_dir ${DATA_DIR} \
	--batch_size ${BATCH_SIZE} \
	--learning_rate ${LR} \
	--seed ${SEED} \
	--source_name ${SOURCE_NAME} \
	--target_name ${TARGET_NAME} \
	--hidden_size ${HIDDEN_SIZE} \
	--rnn_layers ${RNN_LAYERS} \
	--longest_label ${LONGEST_LABEL} \
	--gradient_clip ${GRADIENT_CLIP} \
	--epochs ${EPOCHS} \
	--encoder_attention ${ENCODER_ATTN} \
	--self_attention ${SELF_ATTN}
