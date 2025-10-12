import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

import os
from typing import Tuple

from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, root_path: str, 
                 image_path: str, mask_path: str, 
                 image_size: int, subsample: float = 1.0):
        """
        Create Local Dataset for Image Segmentation

        Args:
            root_path   (str): dataset root path where images and masks directories are present
            image_path  (str): images path (relative to root_path)
            mask_path   (str): masks path (relative to root_path)
            image_size  (int): img_size x img_size to load images
            subsample (float): loads only a subset of images
        """
        self.root_path = root_path
        self.images = sorted([root_path+f"/{image_path}/"+i 
                              for i in os.listdir(root_path+f"/{image_path}/")])
        self.masks = sorted([root_path+f"/{mask_path}/"+i 
                             for i in os.listdir(root_path+f"/{mask_path}/")])

        if len(self.images) != len(self.masks): 
            raise ValueError("Length of images and masks are not the same")
        
        # Subsample if implemented
        self.images = self.images[:int(len(self.images)*subsample)]
        self.masks = self.masks[:int(len(self.masks)*subsample)]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            ])
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images[index]).convert("RGBA")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)