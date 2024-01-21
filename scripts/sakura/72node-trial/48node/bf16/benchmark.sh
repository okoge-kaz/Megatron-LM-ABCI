#!/bin/bash

# 一旦ハードコード

cd /mnt/nfs/home/llmjp0/Megatron-LM/scripts/sakura/72node-trial/48node/bf16

FFN_HIDDEN_SIZE=39936 #39936 #43008
NUM_LAYERS=96
NUM_HEADS=128
NUM_QUERY_GROUPS=16
SEQ_LENGTH=4096
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=8
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1080

# ログファイル名を設定（引数の名前を含む）
LOG_FILE_NAME="llama2-175b-base-selective-mod-by-upa_FFNHS${FFN_HIDDEN_SIZE}_NL${NUM_LAYERS}_NH${NUM_HEADS}_NQG${NUM_QUERY_GROUPS}_SL${SEQ_LENGTH}_TPS${TENSOR_PARALLEL_SIZE}_PPS${PIPELINE_PARALLEL_SIZE}_MBS${MICRO_BATCH_SIZE}_GBS${GLOBAL_BATCH_SIZE}.log"

# train_model.sh スクリプトを実行し、ログに出力
bash llama2-175b-base-selective-mod-by-upa.sh $FFN_HIDDEN_SIZE $NUM_LAYERS $NUM_HEADS $NUM_QUERY_GROUPS \
    $SEQ_LENGTH $TENSOR_PARALLEL_SIZE $PIPELINE_PARALLEL_SIZE $NUM_GPUS $MICRO_BATCH_SIZE $GLOBAL_BATCH_SIZE \
    >> logs/$LOG_FILE_NAME 2>&1
