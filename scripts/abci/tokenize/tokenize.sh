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
cd /bb/llm/gaf51275/llama/Megatron-LM-freeze
source .env/bin/activate

DATASET_DIR=/bb/llm/gaf51275/jalm/JParaCrawl3.0
OUTPUT_DIR=/bb/llm/gaf51275/llama/datasets/JParaCrawl3.0

TOKENIZER_MODEL=/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_okazaki_lab_cc_nfkc_16k_aligned_8/merged_tokenizer_sp/jalm_llama.model

mkdir -p ${OUTPUT_DIR}

# tokenize
INPUT_FILE=${DATASET_DIR}/default_plain_text_format.jsonl

python tools/preprocess_data.py \
  --input ${INPUT_FILE} \
  --output-prefix ${OUTPUT_DIR}/default_plain_text_format \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --append-eod \
  --workers 64

# tokenize
INPUT_FILE=${DATASET_DIR}/highquality_plain_text_format.jsonl

python tools/preprocess_data.py \
  --input ${INPUT_FILE} \
  --output-prefix ${OUTPUT_DIR}/highquality_plain_text_format \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --append-eod \
  --workers 64
