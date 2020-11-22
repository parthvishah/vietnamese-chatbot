#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,k80_4
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=an3056@nyu.edu
#SBATCH --job-name="noattn"
#SBATCH --output=/scratch/an3056/nlp_project/outputs/%j.out

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=an3056
source activate nmt_env

# Set project working directory
PROJECT=/scratch/${NETID}/nlp_project

# Set arguments
STUDY_NAME=nmt_noattn_64b_0.25lr_1gpu #nmt_no_attn_batchsize_learningrate_gpus
SAVE_DIR=${PROJECT}/saved_models
DATA_DIR=${PROJECT}/vietnamese-chatbot/data/interim/iwslt15-en-vn/
PLOT_DIR=${PROJECT}/plots
BATCH_SIZE=64
LR=0.25
SEED=42
SOURCE_NAME='en'
TARGET_NAME='vi'
HIDDEN_SIZE=1024
RNN_LAYERS=2
LONGEST_LABEL=1
GRADIENT_CLIP=0.3
EPOCHS=20

cd ${PROJECT}
python ./vietnamese-chatbot/scripts/train_no_attention_test.py \
	--experiment ${STUDY_NAME} \
	--save_dir ${SAVE_DIR} \
	--data_dir ${DATA_DIR} \
	--plots_dir ${PLOT_DIR} \
	--batch_size ${BATCH_SIZE} \
	--learning_rate ${LR} \
	--seed ${SEED} \
	--source_name ${SOURCE_NAME} \
	--target_name ${TARGET_NAME} \
	--hidden_size ${HIDDEN_SIZE} \
	--rnn_layers ${RNN_LAYERS} \
	--longest_label ${LONGEST_LABEL} \
	--gradient_clip ${GRADIENT_CLIP} \
	--epochs ${EPOCHS}
