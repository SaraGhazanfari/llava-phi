# LLaVA-Phi: Small Multi-Modal Assistant

**LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model** [[Paper](https://arxiv.org/pdf/2401.02330)] <br>

## News
[3/18] We release a new project named Mipha \
**Mipha: A Comprehensive Overhaul of Multimodal Assistant with Small Language Models** [[Paper](https://arxiv.org/abs/2403.06199)] [[Code](https://github.com/zhuyiche/Mipha)]<br>
Our Mipha-3B outperforms many existing 3B MLLMs, including Bunny-3B/MobileVLM-v2, using much less training data. We also analyze the design space of small multimodal models with some new findings. Check out our paper and give it a try! 

## Release
[1/26] Now you can download our [model weight]((#llava-weights)).\
[1/15] Our model and training codes are released. \
[1/5] Our codes are currently undergoing an internal review and will be released shortly (expected next week)


## Contents
- [Install](#install)
- [LLaVA-Phi Weights](#llava-weights)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

1. Clone this repository and navigate to llava-phi folder
```bash
git clone https://github.com/zhuyiche/llava-phi.git
cd llava-phi
```

2. Install Package
```Shell
conda create -n llava_phi python=3.10 -y
conda activate llava_phi
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## LLaVA-Phi Weights
Download model weight at [huggingface](https://huggingface.co/zxmonent/llava-phi)

## Training Curve
The training curve can be found at [wandb](https://wandb.ai/ecnu_/llava-phi/reports/LLaVA-Phi-Training-Logs--Vmlldzo2NTkxMjg1)

## Train

LLaVA-Phi training consists of two stages: (1) feature alignment stage: use [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; 
(2) visual instruction tuning stage: visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.

### Hyperparameters
We use a similar set of hyperparameters as LLaVA-1.5 in both pretraining and finetuning phase.  Both hyperparameters used in pretraining and finetuning are provided below.

1. Pretraining

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: | ---: | ---: | ---: |
| LLaVA-Phi      | 256 | 1e-3 | 1 | 2048 | 0 |

2. Finetuning

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
|----------------| ---: | ---: | ---: | ---: | ---: |
| LLaVA-Phi      | 128 | 2e-5 | 1 | 2048 | 0 |

### Download base checkpoints

Our base model is phi-2. You should download the weights from [here](https://huggingface.co/susnato/phi-2), and change the `--model_name_or_path` in [`get_base_model.sh`](https://github.com/zhuyiche/llava-phi/blob/b7266edc8a90e7b11fa3492491a40cdb8993f831/scripts/llava_phi/get_base_model.sh#L4). <br>
Our vision encoder is ViT-L/14 336px. You should download the weights from [here](https://huggingface.co/openai/clip-vit-large-patch14-336).

### Integrate the model
Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain). <br>

Then, you should integrate phi-2 and ViT-L/14 336px into a single model by running the following script:
```bash
bash ./script/llava_phi/get_base_model.sh
cp ./openai/clip-vit-large-patch14-336/preprocessor_config.json ./base_checkpoints_llava_phi
```

### Pretrain (feature alignment)


```bash
bash ./scripts/llava_phi/pretrain.sh
cp ./openai/clip-vit-large-patch14-336/preprocessor_config.json ./checkpoints/llavaPhi-v0-3b-pretrain
```

### Visual Instruction Tuning

Please refer [here](https://github.com/haotian-liu/LLaVA/blob/9a26bd1435b4ac42c282757f2c16d34226575e96/README.md#visual-instruction-tuning) to prepare the instruction tuning data.

Training script with DeepSpeed ZeRO-3: [`finetune.sh`](https://github.com/zhuyiche/llava-phi/blob/main/scripts/llava_phi/finetune.sh).

```bash
bash ./scripts/llava_phi/finetune.sh
cp ./openai/clip-vit-large-patch14-336/preprocessor_config.json ./checkpoints/llavaPhi-v0-3b-finetune
```

## Evaluation

To ensure the reproducibility, we evaluate the models with greedy decoding.

See [Evaluation.md](https://github.com/zhuyiche/llava-phi/blob/main/docs/Evaluation.md).

## Citation

If you find LLaVA-Phi useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{zhu2024llavaphi,
      title={LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model}, 
      author={Yichen Zhu and Minjie Zhu and Ning Liu and Zhicai Ou and Xiaofeng Mou and Jian Tang},
      year={2024},
      eprint={2401.02330},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@article{zhu2024comprehensive,
  title={A Comprehensive Overhaul of Multimodal Assistant with Small Language Models},
  author={Zhu, Minjie and Zhu, Yichen and Liu, Xin and Liu, Ning and Xu, Zhiyuan and Shen, Chaomin and Peng, Yaxin and Ou, Zhicai and Feng, Feifei and Tang, Jian},
  journal={arXiv preprint arXiv:2403.06199},
  year={2024}
}
```

## Acknowledgement
We build our project based on
- [LLaVA](https://github.com/haotian-liu/LLaVA): an amazing open-sourced project for vision language assistant
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): We use this codebase to finetune Phi model
