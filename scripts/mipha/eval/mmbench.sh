#!/bin/bash

SPLIT="mmbench_dev_20230712"

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_mmbench \
    --model-path $MODELDIR \
    --question-file $VAST/eval/mmbench/$SPLIT.tsv \
    --answers-file $VAST/eval/mmbench/answers/$SPLIT/$model_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

$SCRATCH/code/llava-phi/pytorch-example/python scripts/convert_mmbench_for_submission.py \
    --annotation-file $VAST/eval/mmbench/$SPLIT.tsv \
    --result-dir $VAST/eval/mmbench/answers/$SPLIT \
    --upload-dir $VAST/eval/mmbench/answers_upload/$SPLIT \
    --experiment $model_name