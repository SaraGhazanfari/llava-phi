#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file $VAST/eval/pope/llava_pope_test.jsonl \
    --image-folder /path/to/data/coco/val2014 \
    --answers-file $VAST/eval/pope/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode phi

$SCRATCH/code/llava-phi/pytorch-example/python mipha/eval/eval_pope.py \
    --annotation-dir $VAST/eval/pope/coco \
    --question-file $VAST/eval/pope/llava_pope_test.jsonl \
    --result-file $VAST/eval/pope/answers/$model_name.jsonl