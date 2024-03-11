#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -o outputs/change_tp_pp/
#$ -cwd

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
source .env/bin/activate

# distributed settings
TARGET_TENSOR_PARALLEL_SIZE=1   # fixed
TARGET_PIPELINE_PARALLEL_SIZE=1 # fixed

BASE_TENSOR_PARALLEL_SIZE=2  # Llama-2 7B extended
BASE_PIPELINE_PARALLEL_SIZE=8  # Llama-2 7B extended

# model config
BASE_CHECKPOINT_DIR=/groups/gaf51275/llama/checkpoints/pubmed/llama-2-13b-base-megatron/tp${BASE_TENSOR_PARALLEL_SIZE}-pp${BASE_PIPELINE_PARALLEL_SIZE}
TARGET_CHECKPOINT_DIR=/groups/gaf51275/llama/checkpoints/pubmed/llama-2-13b-base-megatron/tp${TARGET_TENSOR_PARALLEL_SIZE}-pp${TARGET_PIPELINE_PARALLEL_SIZE}

mkdir -p ${TARGET_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-hf/tokenizer.model

# change latest_checkpointed_iteration.txt
ITERATION=1500
echo $ITERATION >${BASE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt

# convert
python tools/checkpoint/util.py \
  --model-type GPT \
  --loader megatron \
  --saver megatron \
  --megatron-path /bb/llm/gaf51275/llama/Megatron-LM \
  --target-tensor-parallel-size ${TARGET_TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${TARGET_PIPELINE_PARALLEL_SIZE} \
  --load-dir ${BASE_CHECKPOINT_DIR} \
  --save-dir ${TARGET_CHECKPOINT_DIR}
