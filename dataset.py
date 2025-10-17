from typing import List
import os

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, 
                    root_path: str, 
                    image_path: str,  
                    mask_path: str,
                    image_size: int,
                    heatmap_path: str = None,
                    heatmap_sizes: List[int] = None,
                    subsample: int = 1.0):
        """
        Creates CustomDataset for YOLOU-Seg++
        (1) During Training, we load the precomputed heatmaps, and create two tensors
        (2) During Inference, we do not load the precomputed heatmaps. 
        Args: 
            root_path (str): root path of the dataset directory (e.g., stacked_segmentation)
            image_size (str): 


        """
                    
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i for i in os.listdir(root_path+f"/{image_path}/")])
        self.heatmaps = sorted([root_path+f"/{heatmap_path}/"+i for i in os.listdir(root_path+f"/{heatmap_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i for i in os.listdir(root_path+f"/{mask_path}/")])
        
        if len(self.images) != len(self.masks) and len(self.masks) != len(self.heatmaps): 
            raise ValueError("Length of images, masks, and heatmaps are not the same")
        
        # Subsample
        self.images = self.images[:int(len(self.images)*subsample)]
        self.heatmaps = self.heatmaps[:int(len(self.heatmaps)*subsample)]
        self.masks = self.masks[:int(len(self.masks)*subsample)]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.image_size = image_size
        self.heatmap_sizes = heatmap_sizes
        
    def __getitem__(self, index) -> torch.tensor:
        img = Image.open(self.images[index]).convert("RGBA") # <- 4-channels (t1c, t1n, t2w, t2f)
        mask = Image.open(self.masks[index]).convert("L")    # <- 1-channel (binary mask)
        img_tensor, mask_tensor = self.transform(img), self.transform(mask)
        
        if self.heatmap_sizes: # TRAINING: if we have heatmap sizes, we load the heatmaps (faster training)
            heatmap = Image.open(self.heatmaps[index]).convert("L")
            
            heatmap_tensors = []
            for size in heatmap_sizes: 
                heatmap_tensor = transforms.ToTensor(
                    transforms.Resize((size, size))(heatmap)
                    )

                # Heatmap normalization
                mean, std  = heatmap_tensor.mean(), heatmap_tensor.std()
                if std > 0: heatmap_tensor = (heatmap_tensor - mean) / std
                else: heatmap_tensor = heatmap_tensor - mean # Avoid NaN if std = 0

                heatmap_tensors.append(heatmap_tensor)
            return img_tensor, mask_tensor, heatmap_tensors
        else:                  # Inference: if we do not have heatmap sizes, we do not load them
            return img_tensor, mask_tensor
            
    def __len__(self) -> int:
        return len(self.images)
