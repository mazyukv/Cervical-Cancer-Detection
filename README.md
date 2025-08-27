# Artificial Intelligence in Cervical Cancer Screening: Comparing CNN-based and Transformer-based Approaches for Abnormal Cell Detection

This repository contains code to reproduce the experiments from my thesis:

1. Re-implement and adapt Faster R-CNN from Zhang et al. (2025)
2. Train ViTDet variants.
3. Evaluate, ablate, and analyze results.

## Repository Structure

```
├── csvfiles/ # csvs with annotations
├── data/ # images
├── network/ # model and backbone code for Faster R-CNN
├── results/ # logs, checkpoints, figures
├── tool/ # tiny helpers and scripts, custom evaluation script for both models
├── vit/ # ViTDet code (train/eval/infer)
│ └── ... # (configs, experiments, scripts and deps set B (ViTDet)(requirements_v3.txt))
├── _utils.py
├── datasets.py # dataset and transform definitions
├── launch.sh # convenience launcher for Faster R-CNN
├── path_changer.py # fixes local image paths if repo is moved
├── train.py # Faster R-CNN training entrypoint
├── trainer.py # Faster R-CNN training loop utilities
├── requirements_v1.txt # deps set A (Faster R-CNN)
└── README.md
```

## Images

The raw images used in this project (≈32 GB) are not stored in the repository due to size limits.  
They can be downloaded from the following link:

[Download Images](https://springernature.figshare.com/articles/dataset/A_large_annotated_cervical_cytology_images_dataset_for_AI_models_to_aid_cervical_cancer_screening/27901206)

After downloading, place them into the `data/JPEGImages/`.

## Setup: Python environment

```bash
# create & activate a venv (recommended)
py -3.9 -m venv .venv
source .venv/bin/activate

# Option A: Faster R-CNN environment
pip install -r requirements_v1.txt

# Option B: ViTDet environment
cd vit
pip install -r requirements_v3.txt
```

## Training and evaluating Faster R-CNN

```bash
source .venv/bin/activate
./launch.sh # trains and evaluates Faster R-CNN
```

## Training and evaluating ViTDet

```bash
source .venv/bin/activate
cd vit
python lazyconfig_train_net.py   --config-file mask_rcnn_vitdet_b_100ep.py   train.output_dir=./output/vitdet_b   model.roi_heads.num_classes=1
# After training
python predictions_evaluation.py
```
