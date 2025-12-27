<div align="center">

<!-- <h1>JiuTian (九天) </h1> -->
<h2 class="papername"> MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models </h2>
<div>
<div>
    <a href="https://www.slywiki.cn/" target="_blank">Leyang Shen*</a>,
    <a href="https://scholar.google.com/citations?user=Mpg0w3cAAAAJ" target="_blank">Gongwei Chen*</a>,
    <a href="https://rshaojimmy.github.io/" target="_blank">Rui Shao†</a>,
    <a href="http://faculty.hitsz.edu.cn/guanweili" target="_blank">Weili Guan</a>,
    <a href="http://faculty.hitsz.edu.cn/guanweili" target="_blank">Liqiang Nie†</a>
</div>

School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen<br>
*Equal contribution
†Corresponding author

[The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)](https://neurips.cc/Conferences/2024)

[[Paper]](https://arxiv.org/abs/2407.12709)
[[Project Page]](https://www.slywiki.cn/mome/)

</div>
<br>

</div>

## If you find this work useful for your research, please kindly cite our paper and star our repo.

## Updates
- [12/2024] Code and checkpoints are released.
- [09/2024] [Project page](https://www.slywiki.cn/mome/) released!
- [09/2024] MoME has been accepted by NeurIPS 2024!
- [07/2024] [Arxiv paper](https://arxiv.org/abs/2407.12709) released.

## Introduction

This is the github repository of *MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models*. In this work, we propose a mixture of multimodal experts (MoME) to mitigate task interference and obtain a generalist MLLM.

Our MoME is composed of two key components, a mixture of vision experts (MoVE) and a mixture of language experts (MoLE). MoVE can adaptively modulate the features transformed from various vision encoders, and has a strong compatibility in transformation architecture. MoLE incorporates sparsely gated experts into LLMs to achieve painless improvements with roughly unchanged inference costs.

The architecture of the proposed MoME model:

<div align="center">
<img src='./assets/MoME-Architecture.jpg' width='100%'>
</div>

## Installation

### Download
```bash
git clone https://github.com/JiuTian-VL/MoME.git
cd MoME
```

### Environment

```bash
conda create -n mome python=3.12
conda activate mome
pip install -r requirements.txt
```

### Checkpoints

Please download all the required checkpoints by running the `download_ckpt.py` script.

```bash
python download_ckpt.py
```

The required checkpoints will be downloaded to the `./checkpoints` directory from huggingface.

## Training

We provide a training script and instruction to perform stage 1 and stage 2 training.

### Preparation

- Download dataset from [huggingface](https://huggingface.co/datasets/daybreaksly/MoME-data-train)
- Download images and organized them in one folder:

Please download the following datasets:

- **Training images**
  - `coco-2014`
  - `coco-2017`
  - `flickr30K`
  - `gqa`
  - `iconqa`
  - `ureader-instruction-1.0`

After downloading, place all these folders under a **single directory**.  
For example:

```bash
/path/to/image_folder/
├── coco/images/train2014/COCO_train2014_*.jpg
├── coco_2017/train2017/*.jpg
├── flickr30K/images/flickr30k-images/*.jpg
├── gqa/images/*.jpg
├── iconqa/iconqa_data/iconqa/train/choose_txt/**/image.png
└── ureader-instruction-1.0/images/*.jpg

```
In config files (e.g. `configs/train_mome_stage1.yaml` and `configs/train_mome_stage2.yaml`), update the `vis_root` path and dataset paths:

```yaml
train_datasets:
  - ann_path: "/path/to/MoME-data-train/General_data.json"
    vis_root: "/path/to/image_folder"
    sample_ratio: 1
  - ann_path: "/path/to/MoME-data-train/REC_data.json"
    vis_root: "/path/to/image_folder"
    sample_ratio: 1
  - ann_path: "/path/to/MoME-data-train/REG_data.json"
    vis_root: "/path/to/image_folder"
    sample_ratio: 1
  - ann_path: "/path/to/MoME-data-train/Doc_data.json"
    vis_root: "/path/to/image_folder"
    sample_ratio: 1
```

### Two-stage Training

1. Run multi‑GPU training for stage 1:

```
cd MoME
bash scripts/train_stage1.sh
```

2. Update the checkpoint path of `stage 1` in `configs/train_mome_stage2.yaml`

```yaml
model:
  ckpt: "outputs/mome_stage1/your_job_id/checkpoint_0.pth"
```

3. Run multi‑GPU training for stage 2:

```
cd MoME
bash scripts/train_stage2.sh
```

## Inference and Demo

We provide an inference example in `playground.ipynb`, which includes a minimal example of how to use the MoME model for inference.

A gradio demo used for model testing and router visualization is also provided in `demo_mome.py`. You can start the demo by running the following command:

```bash
python demo_mome.py
```

## Multitasking Benchmark

We collected 24 datasets and categorized them into four groups for instruction-tuning and evaluation:

<div align="center">
<img src='./assets/MoME-Dataset.jpg' width='100%'>
</div>

## Evaluation results

Here we list the multitasking performance comparison of MoME and baselines. Please refer to our paper for more details.

![MoVE](assets/MoME-MoVE.jpg)
![MoLE](assets/MoME-MoLE.jpg)
![SOTA](assets/MoME-ALL.jpg)

## Qualitative Examples
![Qualitative Examples](assets/MoME-Qualitative.jpg)

## Citation

If you find this work useful for your research, please kindly cite our paper:
```
@inproceedings{shen2024mome,
    title={MoME: Mixture of Multimodal Experts for Generalist Multimodal Large Language Models}, 
    author={Shen, Leyang and Chen, Gongwei and Shao, Rui and Guan, Weili and Nie, Liqiang},
    booktitle={Advances in neural information processing systems},
    year={2024}
}
```
