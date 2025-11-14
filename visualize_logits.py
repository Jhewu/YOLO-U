from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from dataset import CustomDataset
import torch

import os
import time
from typing import Tuple, List
from itertools import cycle

import torch
from torch.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader

from torchinfo import summary
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

import cv2
from torchvision import transforms  
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur  
import torchvision
import torch

import torch.nn.functional as F

def spatial_confidence(logits, k_frac=0.05):
    probs = torch.sigmoid(logits).flatten()
    k = max(1, int(k_frac * probs.numel()))
    topk_mean = probs.topk(k).values.mean()
    return topk_mean

def argmax_conf(detect_branch): 
    class_probs = detect_branch[0, 4, :]        # [525]
    confidence_scores = torch.sigmoid(class_probs)
    idx = confidence_scores.argmax()
    # return detect_branch[0, :4, idx]
    return confidence_scores[idx]

if __name__ == "__main__": 
    # Create trainer and predictor instances
    p_args = dict(model="new_yolo_checkpoint/weights/best.pt",
                data=f"data/data.yaml", 
                verbose=True,
                imgsz=160, 
                save=False)

    # Create predictor and Load checkpoint
    YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
    YOLO_predictor.setup_model(p_args["model"])

    x = cv2.imread("archive/BraTS-SSA-00041-00043-t1c_image.png", cv2.IMREAD_UNCHANGED)
    
    x = transforms.ToTensor()(x)
    x = transforms.Resize(size=(160, 160))(x)
    x = x.to("cuda")  
    x = x.unsqueeze(0)  

    model = YOLO_predictor.model.model
    model.to("cuda")
    x = model(x)

    detect_branch, cls_branch = x
    a,b,c = cls_branch

    """SPLIT CHANNELS
    B, G, R, A = cv2.split(x)
    channels = [B, G, R, A]
    names = ['channel_B_data', 'channel_G_data', 'channel_R_data', 'channel_A_data']
    
    for channel, name in zip(channels, names):
        filename = f'{name}.png'
        success = cv2.imwrite(filename, channel)
    """
    
    conf = argmax_conf(detect_branch)

    heatmap = a[:, -1:]
    heatmap = torch.sigmoid(heatmap)
    confidence = argmax_conf(detect_branch)
    spatial_confidence = spatial_confidence(heatmap)
    
    print(confidence, spatial_confidence)
    plt.imshow(heatmap.squeeze(0).squeeze(0).cpu().numpy())
    plt.show()

"""TRASH CODE CORNER

heatmap = torchvision.transforms.functional.resize(heatmap, size=(160,160), interpolation=transforms.InterpolationMode.BICUBIC)

    # heatmap = GaussianBlur(kernel_size=5)(heatmap)

#     mean = heatmap.mean()
#     std = heatmap.std()
# 
#     if std > 0: heatmap = (heatmap - mean) / std
#     else: heatmap = heatmap - mean # Avoid NaN if std = 0
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    # heatmap = (heatmap > 0.5).float()
# heatmap = torch.sigmoid(heatmap)

"""
