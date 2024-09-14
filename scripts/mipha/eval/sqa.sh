#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_science \
    --model-path $MODELDIR \
    --question-file $VAST/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder $VAST/eval/scienceqa/images/test \
    --answers-file $VAST/eval/scienceqa/answers/$model_name.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v0

$SCRATCH/code/llava-phi/pytorch-example/python mipha/eval/eval_science_qa.py \
    --base-dir $VAST/eval/scienceqa \
    --result-file $VAST/eval/scienceqa/answers/$model_name.jsonl \
    --output-file $VAST/eval/scienceqa/answers/$model_name-output.jsonl \
    --output-result $VAST/eval/scienceqa/answers/$model_name-result.json

