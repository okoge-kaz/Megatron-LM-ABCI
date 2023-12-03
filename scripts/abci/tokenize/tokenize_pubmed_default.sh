#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=15:00:00
#$ -j y
#$ -o outputs/tokenize/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# python virtualenv
source .env/bin/activate

DATASET_DIR=/groups/gcd50698/fujii/datasets/PUBMED_20230517
OUTPUT_DIR=/groups/gcd50698/fujii/datasets/PUBMED_20230517/binarized/llama-tokenizer
mkdir -p ${OUTPUT_DIR}

TOKENIZER_MODEL=/bb/llm/gaf51275/llama/huggingface-checkpoint/Llama-2-7b-hf/tokenizer.model

mkdir -p ${OUTPUT_DIR}

# tokenize
INPUT_FILE=${DATASET_DIR}/PUBMED.jsonl

python tools/preprocess_data.py \
  --input ${INPUT_FILE} \
  --json-keys english japanese \
  --output-prefix ${OUTPUT_DIR}/pubmed_llama_default \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --append-eod \
  --workers 64
