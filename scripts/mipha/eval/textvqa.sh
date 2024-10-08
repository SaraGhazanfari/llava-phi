#!/bin/bash

#model_name=MiphaGemma-v0-2b-finetune
model_name=MiphaPhi2-v0-3b-finetune
SLM=phi_2
VIT=siglip
MODELDIR=./ckpts/checkpoints-$VIT/$SLM/$model_name

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.model_vqa_loader \
    --model-path $MODELDIR \
    --question-file $VAST/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder $VAST/eval/textvqa/train_images \
    --answers-file $VAST/eval/textvqa/answers/$model_name.jsonl \
    --temperature 0 \
    --conv-mode v0

$SCRATCH/code/llava-phi/pytorch-example/python -m mipha.eval.eval_textvqa \
    --annotation-file $VAST/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $VAST/eval/textvqa/answers/$model_name.jsonl
