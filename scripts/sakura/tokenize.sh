#!/bin/bash

set -e

cd /mnt/nfs/home/llmjp0/Megatron-LM
source .env/bin/activate

TOKENZIER="/mnt/nfs/home/llmjp0/llm-jp-tokenizer/models/ver2.2/code20K_en40K_ja60K.ver2.2.model"

python tools/preprocess_data.py \
  --input /mnt/nfs/home/llmjp0/datasets/ja_wiki/ja_wiki_merged.jsonl \
  --output-prefix /mnt/nfs/home/llmjp0/binarized/ver2.2/code20K_en40K_ja60K/ja_wiki/ja_wiki \
  --tokenizer-model $TOKENZIER \
  --tokenizer-type SentencePieceTokenizer \
  --workers 128 \
  --append-eod

echo "done"