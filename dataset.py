from typing import List, Tuple
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
                    objectmap_path: str = None,
                    objectmap_sizes: List[int] = [20, 10],
                    subsample: int = 1.0):
        """
        Creates CustomDataset for YOLO-U
        (1) During Training, we load the precomputed objectmap, and create two tensors
        (2) During Inference, we do not load the precomputed objectmap
        """
        self.root_path = root_path
        self.image_dir = root_path + f"/{image_path}/"
        self.mask_dir = root_path + f"/{mask_path}/"
        self.objectmap_dir = root_path + f"/{objectmap_path}/" if objectmap_path else None
        
        # We only list image files and use their names to derive mask/heatmap paths
        image_filenames = sorted(os.listdir(self.image_dir))
        
        # Save only the basename without extension (e.g., 'image001')
        self.basenames = [os.path.splitext(f)[0] for f in image_filenames]
        
        # Subsample the basenames
        self.basenames = self.basenames[:int(len(self.basenames) * subsample)]
        
        # Basic validation check (You may want to add more robust checks)
        # Check if corresponding mask files exist
        for basename in self.basenames:
            if not os.path.exists(self.mask_dir + basename + ".png"): # Assuming masks are .png
                 raise FileNotFoundError(f"Mask file not found for {basename}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.image_size = image_size
        self.objectmap_sizes = objectmap_sizes
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        
        basename = self.basenames[index]
        
        # 1. Load Image and Mask
        img_path = self.image_dir + basename + ".png" # Assuming image extension
        mask_path = self.mask_dir + basename + ".png" # Assuming mask extension
        
        # The original code used .convert("RGBA") and .convert("L"). We keep it.
        img = Image.open(img_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")
        
        img_tensor, mask_tensor = self.transform(img), self.transform(mask)
        
        if self.objectmap_sizes and self.objectmap_dir: # TRAINING: Load pre-saved heatmaps
            objectmap_tensors = []
            
            for size in self.objectmap_sizes: 
                # Construct the pre-saved heatmap filename: filename_size.png
                objectmap_filename = f"{basename}_{size}.png"
                objectmap_path = self.objectmap_dir + objectmap_filename
                
                # Load the pre-resized heatmap
                objectmap = Image.open(objectmap_path).convert("L")
                
                # Apply only ToTensor, as the image is already the correct size (size x size)
                objectmap_tensor = transforms.ToTensor()(objectmap) 
                
                objectmap_tensors.append(objectmap_tensor)
                
            return img_tensor, mask_tensor, objectmap_tensors
        else: # Inference: no heatmaps
            return img_tensor, mask_tensor
            
    def __len__(self) -> int:
        return len(self.basenames)
