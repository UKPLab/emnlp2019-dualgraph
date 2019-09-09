#!/bin/bash

if [ "$#" -lt 5 ]; then
  echo "./train_LDC2017T10.sh <gpuid> <gnn_type> <gnn_layers> <start_decay_steps> <decay_steps>"
  exit 2
fi

mkdir -p data/models

GPUID=$1
GNNTYPE=$2
GNNLAYERS=$3
STARTDECAYSTEPS=$4
DECAYSTEPS=$5
DATASET=ldc2017t10
STEPS=1830
EPOCHS=$((${STEPS}*60))

export CUDA_VISIBLE_DEVICES=${GPUID}
export OMP_NUM_THREADS=10

python -u train.py -data data/${DATASET} \
-save_model data/models/${DATASET}-${GNNTYPE} \
-rnn_size 900 -word_vec_size 300 -train_steps ${EPOCHS} -optim adam \
-valid_steps ${STEPS} \
-valid_batch_size 1 \
-encoder_type graph \
-gnn_type ${GNNTYPE} \
-gnn_layers ${GNNLAYERS} \
-decoder_type rnn \
-learning_rate 0.001 \
-dropout 0.4 \
-copy_attn -copy_attn_type mlp -coverage_attn -batch_size 20 \
-save_checkpoint_steps ${STEPS} \
-start_decay_steps ${STARTDECAYSTEPS} \
-decay_steps ${DECAYSTEPS} \
-layers 2 \
-global_attention mlp \
-pre_word_vecs_enc data/${DATASET}.embeddings.enc.pt \
-pre_word_vecs_dec data/${DATASET}.embeddings.dec.pt \
-gpu_ranks 0
