"""
This file runs inference and evaluation for a trained ViTDet-based Mask R-CNN model on the TCT test dataset. 
It loads the trained model from checkpoint, applies it to each test image, filters predictions by a confidence 
threshold, and saves results (image paths and predicted boxes) into a CSV file. After prediction, it evaluates 
the results against ground truth annotations using a custom VOC-style evaluation function, reporting AP, mAP, 
and F1-score metrics.
"""
import csv
import torch
from mask_rcnn_vitdet_b_100ep import load_tct_dataset
from detectron2.config import LazyConfig, instantiate
import cv2
import sys
import os
import pandas as pd
from PIL import Image
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.voc_eval_new_vit import custom_voc_eval

# === Load config and model ===
cfg = LazyConfig.load("C:/Users/Asus/Desktop/Cervical-Cancer-Detection/results/vit/output/vitdet_b/config.yaml")
cfg.train.init_checkpoint = "C:/Users/Asus/Desktop/Cervical-Cancer-Detection/results/vit/output/vitdet_b/model_final.pth"
cfg.model.roi_heads.box_predictor.test_score_thresh = 0.7  # match threshold

model = instantiate(cfg.model)
model.eval()
model.to("cpu")  # CPU inference
weights = torch.load(cfg.train.init_checkpoint, map_location="cpu")
model.load_state_dict(weights["model"])

# === Load dataset slice ===
dataset = load_tct_dataset("C:/Users/Asus/Desktop/Cervical-Cancer-Detection/csvfiles/test.csv")

# === Output CSV ===
out_csv_path = "C:/Users/Asus/Desktop/Cervical-Cancer-Detection/results/vit/output/predictions.csv"
with open(out_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "prediction"])

    for sample in dataset:
        image_path = sample["file_name"]

        # Load image same as single-image script
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Prepare tensor input
        inputs = {"image": torch.from_numpy(image_np.transpose(2, 0, 1)).float().to("cpu")}
        
        # Run prediction
        with torch.no_grad():
            outputs = model([inputs])[0]

        pred_instances = outputs["instances"].to("cpu")
        boxes = pred_instances.pred_boxes.tensor.numpy()
        scores = pred_instances.scores.numpy()

        preds_str_list = []
        for i in range(len(boxes)):
            score = scores[i]
            if score >= 0.7:  # ✅ filter
                x1, y1, x2, y2 = boxes[i]
                preds_str_list.append(f"1 {score} {x1} {y1} {x2} {y2}")

        pred_str = ";".join(preds_str_list)
        writer.writerow([image_path, pred_str])

print(f"✅ Predictions saved to {out_csv_path}")

testap_dict,test_mAP,testmf1 = custom_voc_eval(
    gt_csv="C:/Users/Asus/Desktop/Cervical-Cancer-Detection/csvfiles/test.csv",
    pred_csv=out_csv_path,
    tp_save_path='C:/Users/Asus/Desktop/Cervical-Cancer-Detection/results/vit/output/tp_images.txt'
) 
print(f"Test TCT AP :{testap_dict['1']:.4f}")
print(f"Test mAP: {test_mAP:.4f}")
print(f"Test F1-score: {testmf1:.4f}")
