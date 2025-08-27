"""
This file is responsible for loading our custom cervical cancer dataset 
and specifying the training configurations for a ViTDet-based in Detectron2. 
It registers the training and validation splits from CSV annotations, 
converting them into Detectron2’s dataset format with a single class ("cancer"). 
In addition, it defines the model initialization, optimizer, learning rate schedule, and dataloader settings, 
providing the full configuration needed to run experiments on our dataset.

"""
from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, get_detection_dataset_dicts
from detectron2.data import transforms as T


from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import os
import pandas as pd
from PIL import Image

def load_tct_dataset(csv_path):
    df = pd.read_csv(csv_path)
    dataset_dicts = []

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path)
        width, height = image.size

        record = {
            "file_name": img_path,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": [],
        }

        for ann in str(row["annotation"]).split(";"):
            if not ann.strip():
                continue
            parts = ann.strip().split()
            if len(parts) != 5:
                continue
            category_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:])
            record["annotations"].append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": category_id,
            })

        dataset_dicts.append(record)

    return dataset_dicts

# Register both train and val datasets
DatasetCatalog.register("tct_train", lambda: load_tct_dataset("C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/csvfiles/fold1/train.csv"))
MetadataCatalog.get("tct_train").set(thing_classes=["cancer"])

DatasetCatalog.register("tct_val", lambda: load_tct_dataset("C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/csvfiles/fold1/val.csv"))
MetadataCatalog.get("tct_val").set(thing_classes=["cancer"])


model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
)


# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 111560

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[100404, 105982],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,  # or use 0.00224 directly
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}


dataloader = dict(
    train=L(build_detection_train_loader)(
        dataset=L(get_detection_dataset_dicts)(names=["tct_train"]),
        mapper=L(DatasetMapper)(
            is_train=True,
            augmentations=[
            L(T.ResizeShortestEdge)(
            short_edge_length=(1024, 1024),
            max_size=1024,
            sample_style="choice"
        )
    ],  # Add real augmentations later if needed
            image_format="BGR",
        ),
        total_batch_size=2,
    ),
    test=L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names=["tct_val"]),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[],
            image_format="BGR",
        ),
    ),
)