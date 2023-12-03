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
OUTPUT_DIR=/groups/gcd50698/fujii/datasets/PUBMED_20230517/binarized/okazaki_lab
mkdir -p ${OUTPUT_DIR}

TOKENIZER_MODEL=/bb/llm/gaf51275/jalm/jalm-tokenizer-private/tokenizer/jalm_llama_okazaki_lab_cc_nfkc_16k_aligned_8/merged_tokenizer_sp/jalm_llama.model

mkdir -p ${OUTPUT_DIR}

# tokenize
INPUT_FILE=${DATASET_DIR}/PUBMED.jsonl

python tools/preprocess_data.py \
  --input ${INPUT_FILE} \
  --json-keys english japanese \
  --output-prefix ${OUTPUT_DIR}/pubmed_llama_okazaki_lab \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --append-eod \
  --workers 64
