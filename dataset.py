from typing import List, Tuple
import os
import torch
import numpy as np
import cv2 # <-- Import OpenCV
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from PIL import Image # <-- Removed

class CustomDataset(Dataset):
    def __init__(self,
                    root_path: str,
                    image_path: str,
                    mask_path: str,
                    image_size: int,
                    objectmap_path: str = None,
                    objectmap_sizes: List[int] = [20, 10],
                    subsample: float = 1.0):

        self.root_path = root_path
        self.image_dir = root_path + f"/{image_path}/"
        self.mask_dir = root_path + f"/{mask_path}/"
        self.objectmap_dir = root_path + f"/{objectmap_path}/" if objectmap_path else None

        image_filenames = sorted(os.listdir(self.image_dir))
        self.basenames = [os.path.splitext(f)[0] for f in image_filenames]
        self.basenames = self.basenames[:int(len(self.basenames) * subsample)]

        # Basic validation check (You may want to add more robust checks)
        for basename in self.basenames:
            if not os.path.exists(self.mask_dir + basename + ".png"):
                raise FileNotFoundError(f"Mask file not found for {basename}")

        self.image_size = image_size
        self.objectmap_sizes = objectmap_sizes

        # Note: transforms.ToTensor() handles the HWC -> CWH and /255 conversion
        # We replace transforms.Resize(PIL image) with cv2.resize (numpy array) inside __getitem__
        self.to_tensor = transforms.ToTensor()
        self.image_size = image_size
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        basename = self.basenames[index]

        # 1. Construct Paths
        img_path = self.image_dir + basename + ".png"
        mask_path = self.mask_dir + basename + ".png"

        # 2. Load Image and Mask using cv2 (Faster I/O to numpy array)
        
        # Load Image (RGBA -> BGRA in cv2.IMREAD_UNCHANGED)
        # Assuming your input images are 4-channel PNGs (RGBA)
        img_rgb_a = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
        # Load Mask (L/Grayscale)
        # cv2.IMREAD_GRAYSCALE loads as a 2D array (H, W)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 3. Resize using cv2 (Faster than PIL resize)
        interpolation = cv2.INTER_LINEAR # or INTER_CUBIC
        
        # Resize Image
        img_resized = cv2.resize(img_rgb_a, (self.image_size, self.image_size), interpolation=interpolation)
        # Resize Mask (use nearest neighbor for discrete masks)
        mask_resized = cv2.resize(mask_np, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # 4. Convert to PyTorch Tensor (HWC -> CWH, /255)
        img_tensor = self.to_tensor(img_resized)
        # Masks often need to be Float Tensors with an extra dimension [1, H, W]
        mask_tensor = self.to_tensor(mask_resized) 

        # 5. Load Objectmaps (Similar conversion logic)
        if self.objectmap_sizes and self.objectmap_dir:
            objectmap_tensors = []
            
            # Construct the pre-saved heatmap filename: filename_size.png
            objectmap_filename = f"{basename}_{20}.pt"
            objectmap_path = self.objectmap_dir + objectmap_filename
            
            # Load the pre-resized objectmap (L/Grayscale)
            # objectmap_np = cv2.imread(objectmap_path, cv2.IMREAD_GRAYSCALE)
            
            # Since the objectmap is assumed to be the correct size, just convert to tensor.
            # Convert to [1, H, W] tensor
            # objectmap_tensor = self.to_tensor(objectmap_np)
            objectmap_tensor = torch.load(objectmap_path).squeeze(0)

            # Normalization
            mean = objectmap_tensor.mean()
            std = objectmap_tensor.std()

            if std > 0: objectmap_tensor = (objectmap_tensor - mean) / std
            else: objectmap_tensor = objectmap_tensor - mean # Avoid NaN if std = 0
            
            # objectmap_tensors.append(objectmap_tensor)
            
            return img_tensor, mask_tensor, torch.sigmoid(objectmap_tensor)
            # return img_tensor, mask_tensor, objectmap_tensor
        else: # Inference: no heatmaps
            return img_tensor, mask_tensor
            
    def __len__(self) -> int:
        return len(self.basenames)

# from typing import List, Tuple
# import os
# import torch
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from torch.utils.data.dataset import Dataset
# 
# class CustomDataset(Dataset):
#     def __init__(self, 
#                     root_path: str, 
#                     image_path: str, 
#                     mask_path: str,
#                     image_size: int,
#                     objectmap_path: str = None,
#                     objectmap_sizes: List[int] = [20, 10],
#                     subsample: int = 1.0):
#         """
#         Creates CustomDataset for YOLO-U
#         (1) During Training, we load the precomputed objectmap, and create two tensors
#         (2) During Inference, we do not load the precomputed objectmap
#         """
#         self.root_path = root_path
#         self.image_dir = root_path + f"/{image_path}/"
#         self.mask_dir = root_path + f"/{mask_path}/"
#         self.objectmap_dir = root_path + f"/{objectmap_path}/" if objectmap_path else None
#         
#         # We only list image files and use their names to derive mask/heatmap paths
#         image_filenames = sorted(os.listdir(self.image_dir))
#         
#         # Save only the basename without extension (e.g., 'image001')
#         self.basenames = [os.path.splitext(f)[0] for f in image_filenames]
#         
#         # Subsample the basenames
#         self.basenames = self.basenames[:int(len(self.basenames) * subsample)]
#         
#         # Basic validation check (You may want to add more robust checks)
#         # Check if corresponding mask files exist
#         for basename in self.basenames:
#             if not os.path.exists(self.mask_dir + basename + ".png"): # Assuming masks are .png
#                  raise FileNotFoundError(f"Mask file not found for {basename}")
# 
#         self.transform = transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#         ])
#         self.image_size = image_size
#         self.objectmap_sizes = objectmap_sizes
#         
#     def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
#         basename = self.basenames[index]
#         
#         # 1. Load Image and Mask
#         img_path = self.image_dir + basename + ".png" # Assuming image extension
#         mask_path = self.mask_dir + basename + ".png" # Assuming mask extension
#         
#         # The original code used .convert("RGBA") and .convert("L"). We keep it.
#         img = Image.open(img_path).convert("RGBA")
#         mask = Image.open(mask_path).convert("L")
#         
#         img_tensor, mask_tensor = self.transform(img), self.transform(mask)
#         
#         if self.objectmap_sizes and self.objectmap_dir: # TRAINING: Load pre-saved heatmaps
#             objectmap_tensors = []
#             
#             # for size in self.objectmap_sizes: 
#             # Construct the pre-saved heatmap filename: filename_size.png
#             objectmap_filename = f"{basename}_{20}.png"
#             objectmap_path = self.objectmap_dir + objectmap_filename
#             
#             # Load the pre-resized heatmap
#             objectmap = Image.open(objectmap_path).convert("L")
#             
#             # Apply only ToTensor, as the image is already the correct size (size x size)
#             objectmap_tensor = transforms.ToTensor()(objectmap) 
#             
#             objectmap_tensors.append(objectmap_tensor)
#                 
#             return img_tensor, mask_tensor, objectmap_tensors
#         else: # Inference: no heatmaps
#             return img_tensor, mask_tensor
#             
#     def __len__(self) -> int:
#         return len(self.basenames)
