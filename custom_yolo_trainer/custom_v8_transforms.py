from ultralytics.data.augment import (
    Mosaic, RandomPerspective, LetterBox, 
    Compose, CopyPaste, MixUp, CutMix, 
    Albumentations, RandomHSV, RandomFlip, )
from ultralytics.utils import LOGGER, IterableSimpleNamespace
import numpy as np
import cv2

class GaussianNoisePerChannel:
    def __init__(self, p=0.25, noise_std_range=(0.01, 0.05)):
        self.p = p
        self.noise_std_range = noise_std_range

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]  # Shape: (H, W, C)
        
        # Apply different noise std per channel
        noise_std = np.random.uniform(*self.noise_std_range, size=img.shape[-1])
        
        # Add Gaussian noise per channel
        noise = np.random.normal(0, noise_std, img.shape).astype(img.dtype)
        img = np.clip(img + noise, 0, 1)

        labels["img"] = img
        return labels

class RandomResolution:
    def __init__(self, p=0.15, scale_range=(0.6, 0.9)):
        self.p = p
        self.scale_range = scale_range

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]
        scale = np.random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)

        # Downscale
        img_low = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Upscale back
        img = cv2.resize(img_low, (w, h), interpolation=cv2.INTER_LINEAR)

        labels["img"] = img
        return labels
    
class MildGaussianBlur:
    def __init__(self, p=0.15, kernel_size=3, sigma_range=(0.5, 1.5)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]
        sigma = np.random.uniform(*self.sigma_range)
        
        # Apply Gaussian blur per channel
        for c in range(img.shape[2]):
            img[:, :, c] = cv2.GaussianBlur(img[:, :, c], (self.kernel_size, self.kernel_size), sigma)
        
        labels["img"] = img
        return labels
    
class RandomBiasField:
    def __init__(self, p=0.15, alpha_range=(0.1, 0.3), smoothness=0.3):
        self.p = p
        self.alpha_range = alpha_range
        self.smoothness = smoothness  # Controls how smooth the field is

    def __call__(self, labels):
        if np.random.rand() > self.p:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]
        
        # Random center position (-1 to 1 range)
        center_x = np.random.uniform(-1, 1)
        center_y = np.random.uniform(-1, 1)
        
        # Random alpha strength
        alpha = np.random.uniform(*self.alpha_range)
        
        # Random elliptical shape (different scaling for x and y)
        scale_x = np.random.uniform(0.5, 2.0)
        scale_y = np.random.uniform(0.5, 2.0)
        
        # Create coordinate grids
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Shift coordinates to random center
        X_shifted = (X - center_x) * scale_x
        Y_shifted = (Y - center_y) * scale_y
        
        # Create smoother elliptical bias field with Gaussian smoothing
        # Use a more gradual transition
        distance_squared = X_shifted**2 + Y_shifted**2
        
        # Apply smoother bias field - reduce sharpness
        if self.smoothness > 0:
            # Apply Gaussian kernel to smooth the field
            bias = 1 + alpha * np.exp(-distance_squared / (2 * self.smoothness**2))
        else:
            # Original quadratic (but with better bounds)
            bias = 1 + alpha * distance_squared
        
        # Randomly invert the bias field (simulates different coil effects)
        if np.random.rand() > 0.5:
            bias = 2 - bias  # Invert: now stronger at center, weaker at edges
            
        # Clip to reasonable range
        bias = np.clip(bias, 0.5, 1.5)
        
        # Apply to all channels
        img = img * bias[..., None]
        img = np.clip(img, 0, 1)

        labels["img"] = img
        return labels

def v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace, stretch: bool = False):
    """
    Apply a series of image transformations for training.

    This function creates a composition of image augmentation techniques to prepare images for YOLO training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.

    Args:
        dataset (Dataset): The dataset object containing image data and annotations.
        imgsz (int): The target image size for resizing.
        hyp (IterableSimpleNamespace): A dictionary of hyperparameters controlling various aspects of the transformations.
        stretch (bool): If True, applies stretching to the image. If False, uses LetterBox resizing.

    Returns:
        (Compose): A composition of image transformations to be applied to the dataset.

    Examples:
        >>> from ultralytics.data.dataset import YOLODataset
        >>> from ultralytics.utils import IterableSimpleNamespace
        >>> dataset = YOLODataset(img_path="path/to/images", imgsz=640)
        >>> hyp = IterableSimpleNamespace(mosaic=1.0, copy_paste=0.5, degrees=10.0, translate=0.2, scale=0.9)
        >>> transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data = transforms(dataset[0])
    """
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    )

    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0  # both fliplr and flipud require flip_idx
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations.")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            # RandomResolution(),
            GaussianNoisePerChannel(), 
            MildGaussianBlur(), 
            RandomBiasField(),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms
