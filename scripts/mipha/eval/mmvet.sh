#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa \
    --model-path $MODELDIR \
    --question-file $VAST/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder $VAST/eval/mm-vet/images \
    --answers-file $VAST/eval/mm-vet/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p $VAST/eval/mm-vet/results

$SCRATCH/code/llava-phi/pytorch-example/python scripts/convert_mmvet_for_eval.py \
    --src $VAST/eval/mm-vet/answers/$model_name.jsonl \
    --dst $VAST/eval/mm-vet/results/$model_name.json

