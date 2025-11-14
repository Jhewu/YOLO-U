from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from dataset import CustomDataset

from ultralytics.nn.modules import Conv
from torch import nn

import os
import time
from typing import Tuple, List, Union
from shutil import copy
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
from monai.metrics import DiceMetric, HausdorffDistanceMetric

class Trainer: 
    def __init__(self,
                model: YOLOSegPlusPlus,
                data_path: str, 
                model_path: str = None,
                device: str = "cuda",
                early_stopping_start: int = 50,
                image_size: int = 160, 
                batch_size: int = 64, 
                lr: float = 1e-3,
                epochs: int = 100, 
                patience: int = 25,
                load_and_train: bool = False,
                early_stopping: bool = True,
                mixed_precision: bool = True,
                ):
        """
        Initialize the YOLOU Trainer for training and evaluating YOLOU models.
        
        This class handles the complete training loop for YOLOU models including
        data loading, model training, validation, and optional early stopping.
        
        Args:
            model (YOLOU): The YOLOU model instance to be trained
            yolo (CustomSegmentationPredictor): YOLO predictor for inference and evaluation
            data_path (str): Path to the dataset directory containing training data
            model_path (str, optional): Path to pre-trained YOLOv12-Seg model weights to load. Defaults to None.
            device (str, optional): Device to run training on ('cuda' or 'cpu'). Defaults to "cuda".
            early_stopping_start (int, optional): Epoch number to start early stopping monitoring. Defaults to 50.
            image_size (int, optional): Input image size for model training. Defaults to 160.
            batch_size (int, optional): Batch size for training. Defaults to 128.
            lr (float, optional): Learning rate for optimizer. Defaults to 1e-4.
            epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            patience (int, optional): Number of epochs to wait before early stopping. Defaults to 25.
            load_and_train (bool, optional): Whether to load existing model and continue training. Defaults to False.
            early_stopping (bool, optional): Whether to enable early stopping mechanism. Defaults to True.
            mixed_precision (bool, optional): Whether to use mixed precision training. Defaults to True.
         
        Attributes:
            model (YOLOU): The YOLOU model being trained
            yolo (CustomSegmentationPredictor): YOLO predictor for inference
            device (str): Training device ('cuda' or 'cpu')
            data_path (str): Path to training dataset
            model_path (str): Path to pre-trained model weights
            loss (YOLOULoss): Loss function for training
            image_size (int): Input image dimensions
            batch_size (int): Training batch size
            lr (float): Learning rate
            epochs (int): Maximum training epochs
            early_stopping_start (int): Early stopping initiation epoch
            patience (int): Early stopping patience parameter
            load_and_train (bool): Flag for loading and continuing training
            early_stopping (bool): Flag for enabling early stopping
            mixed_precision (bool): Flag for mixed precision training
            history (None): Training history storage (initialized as None)
            
        Methods:
            train: Execute the training process.
            create_dataloader: Get train and validation datasets as dataloaders.
            plot_loss_curves: Visualize training and validation loss/metric curves.
            save_model: Save model training checkpoints.
            get_current_time: Generate a timestamp for logging/output directories.
            create_dir: Create output directories for logs, models, or results.

        """
    
        self.model = model
        self.device = device
        self.data_path = data_path
        self.model_path = model_path
                
        self.loss = DiceLoss(
            include_background = False, # single class
            to_onehot_y = False,        # single class
            sigmoid=True, 
            soft_label = True,          # should improve convergence    
            batch = True,               # should improve stability during training
            reduction="mean")

        self.dice_metric = DiceMetric(
            include_background = False, 
            reduction="mean_batch",     
            get_not_nans = False, 
            ignore_empty = False, 
            num_classes = 2,            # 2 stands for [0, 1], technically single class
            return_with_label = False
        )
        self.hd95 = HausdorffDistanceMetric(
            include_background = False, 
            percentile=95, 
            reduction = "none",
            get_not_nans = True, 
        )

        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_start = early_stopping_start
        self.patience = patience

        # bool
        self.load_and_train = load_and_train
        self.early_stopping = early_stopping
        self.mixed_precision = mixed_precision

        ### non-parameters
        self.history = None

    def get_current_time(self) -> str: 
        """
        Get current time in YMD | HMS format
        Used for creating non-conflicting result dirs
        Returns
            (str) Time in Ymd | HMs format
        """
        current_time = time.localtime()
        return time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    def create_dir(self, directory: str):
        """
        Creates the given directory if it does not exists
        Args: 
            directory (str): directory to be created
        """
        if not os.path.exists(directory):
            os.makedirs(directory) 

    def plot_loss_curves(self, save_path: str, filename: str = "plot.png") -> None:
        """
        Plot every metric stored in 'self.history'.
        The method automatically discovers keys, assigns a colour, and
        draws a legend entry for each.

        Parameters
            save_path (str): Directory to which the plot PNG will be written.
            filename (str):  File name for the saved image (Default "plot.png")
        """
        if not hasattr(self, "history") or not isinstance(self.history, dict):
            raise ValueError("`self.history` must be a dict of metric lists")

        # Create output dir if it does not exist
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(10, 6))

        # Pick a colour palette – reuse if more metrics than colours
        colour_cycle = cycle(
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
             "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
             "#bcbd22", "#17becf"]
        )

        # Sort keys to keep a deterministic order
        for key in sorted(self.history.keys()):
            values: List[float] = self.history[key]
            # Use the key itself as the label (nice formatting optional)
            label = key.replace("_", " ").title()
            plt.plot(values, label=label, color=next(colour_cycle))

        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        out_file = os.path.join(save_path, filename)
        plt.savefig(out_file)
        plt.show()
            
    def create_dataloader(self, data_path: str) -> Tuple[DataLoader, DataLoader]: 
        """
        Create dataloader from CustomDataset
        Depends on SegmentationDataset

        Args:
            data_path (str): root directory of dataset

        Returns:
            (Tuple[Dataloader]): train_dataloader and val_dataloader
        """
        train_dataset = CustomDataset(
                            root_path = data_path, 
                            image_path = "images/train", 
                            objectmap_path = "objectmap/train", 
                            # heatmap_path = "heatmap/train", 
                            mask_path = "masks/train",
                            image_size = self.image_size,
                            # heatmap_sizes = [20])
                            objectmap_sizes = [20])
                            
        val_dataset = CustomDataset(
                            root_path = data_path, 
                            image_path = "images/val", 
                            objectmap_path = "objectmap/val",
                            # heatmap_path = "heatmap/val", 
                            mask_path = "masks/val", 
                            image_size = self.image_size, 
                            # heatmap_sizes = [20])
                            objectmap_sizes = [20])

        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False, 
                                    num_workers=10)
                                    
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False, 
                                    num_workers=10) # <- do not shuffle
                                    
        return train_dataloader, val_dataloader

    
    def train(self) -> None: 
        """
        Trains YOLOU-Seg++ model

        TODO: ADD DESCRIPTION LATER
        """

        # Add model to device
        self.model.to(self.device)  
        self.model.train()

        # Creates the dataloader
        train_dataloader, val_dataloader = self.create_dataloader(data_path=self.data_path)

        # Model training config
        trainable_param = (
            param
            for name, param in self.model.named_parameters()
            if not name.startswith("encoder.")
        )
        trainable_name = (
            name
            for name, param in self.model.named_parameters()
            if not name.startswith("encoder.")
        )

        optimizer = optim.AdamW(trainable_param, lr=self.lr)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = GradScaler(self.device) # --> mixed precision

        # Initialize variables for callbacks
        self.history = dict(train_loss=[], val_loss=[], train_dice_metric=[], val_dice_metric=[], val_hd95_metric=[], val_precision=[], val_recall=[])
        best_val_dice_metric = float("-inf")

        # Create result directory
        dest_dir = f"runs/{self.get_current_time()}" 
        model_dir = os.path.join(dest_dir, "weights")
        self.create_dir(model_dir)

        # Copy model file to destination
        copy("YOLOSegPlusPlus.py", os.path.join(dest_dir, "YOLOSegPlusPlus.py"))

        # Add seed for reproducibility
        SEED = 42

        # 3. Set PyTorch CPU and GPU seeds
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            # Sets seed for all available GPUs
            torch.cuda.manual_seed_all(SEED)

        patience = 0 # --> local patience for early stopping
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            self.dice_metric.reset()
        
            train_start_time = time.time()
            train_running_loss = 0
            train_running_dice_metric = 0

            if self.mixed_precision:
                for idx, img_mask_heatmap in enumerate(tqdm(train_dataloader)):
                    img = img_mask_heatmap[0].float().to(self.device)
                    mask = img_mask_heatmap[1].float().to(self.device)
                    heatmaps = img_mask_heatmap[2].float().to(self.device)
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=self.device): 
                        pred = self.model(img, heatmaps)
                        loss = self.loss(pred, mask)
                    
                    if torch.isnan(loss):
                        print("NaN loss detected!")
                        print("Pred min/max:", pred.min().item(), pred.max().item())
                        print("Mask min/max:", mask.min().item(), mask.max().item())
                        break
                    
                    scaler.scale(loss).backward()

                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(trainable_param, max_norm=1.0)

                    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                    #   although it still skips optimizer.step() if the gradients contain infs or NaNs.
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()

                    # Accumulate loss and metrics
                    train_running_loss += loss.item()

                    # Update metrics
                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary  = (pred_sigmoid > 0.5).float()
                    self.dice_metric(pred_binary, mask)

            else:
                for idx, img_mask_heatmap in enumerate(tqdm(train_dataloader)):
                    img = img_mask_heatmap[0].float().to(self.device)
                    mask = img_mask_heatmap[1].float().to(self.device)
                    heatmaps = img_mask_heatmap[2].float().to(self.device)
                 
                    optimizer.zero_grad()
                    pred = self.model(img, heatmaps)
                    loss = self.loss(pred, mask)

                    train_running_loss += loss.item()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(trainable_param, max_norm=1.0)
                    optimizer.step()

                    # Update metrics
                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary  = (pred_sigmoid > 0.5).float()
                    self.dice_metric(pred_binary, mask)
    
            train_end_time = time.time()
            train_loss = train_running_loss / (idx + 1)
            train_dice_metric = self.dice_metric.aggregate().item()
            
            self.dice_metric.reset() # <- Reset again
            self.hd95.reset()        # <- Reset again
            
            val_running_loss = val_precision = val_recall = 0
            val_start_time = time.time()
            self.model.eval()
            with torch.no_grad():
                for idx, img_mask_heatmap in enumerate(tqdm(val_dataloader)):
                    img = img_mask_heatmap[0].float().to(self.device)
                    mask = img_mask_heatmap[1].float().to(self.device)
                    
                    heatmaps = img_mask_heatmap[2].float().to(self.device)
                    
                    pred = self.model(img, heatmaps)
                    loss = self.loss(pred, mask)

                    # Accumulate Loss and Metrics
                    val_running_loss += loss.item()
                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary  = (pred_sigmoid > 0.5).float()
                    self.dice_metric(pred_binary, mask)   

                    # Calculate Precision and Recall
                    TP = (pred_binary * mask).sum().float()
                    FP = (pred_binary * (1 - mask)).sum().float()
                    FN = ((1 - pred_binary) * mask).sum().float()
                    val_precision+=TP / (TP + FP + 1e-6).item()
                    val_recall+=TP / (TP + FN + 1e-6).item()

                    # Calculate HD95
                    pred_hot_encoded = torch.cat([1 - pred_binary, pred_binary], dim=1)
                    mask_hot_encoded = torch.cat([1 - mask, mask], dim=1)                    
                    self.hd95(pred_hot_encoded, mask_hot_encoded)

                val_loss = val_running_loss / (idx + 1)
                val_precision = (val_precision / (idx + 1)).item()
                val_recall = (val_recall / (idx + 1)).item()
                val_dice_metric = self.dice_metric.aggregate().item()

                # HD95 Metric
                hd95_raw_results, hd95_not_nans_count = self.hd95.aggregate()
                is_valid = hd95_not_nans_count.bool()
                successful_hd95_values = hd95_raw_results[is_valid]
                val_hd95_metric = torch.mean(successful_hd95_values).item()

            val_end_time = time.time()
            # Update the scheduler
            scheduler.step()
            
            # update the history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_dice_metric"].append(train_dice_metric)
            self.history["val_dice_metric"].append(val_dice_metric)
            self.history["val_hd95_metric"].append(val_hd95_metric)
            self.history["val_precision"].append(val_precision)
            self.history["val_recall"].append(val_recall)
                        
            if val_dice_metric > best_val_dice_metric: 
                if abs(best_val_dice_metric - val_dice_metric) > 1e-3:
                    print(f"Validation Dice Metric improved from {best_val_dice_metric:.4f} to {val_dice_metric:.4f}. Saving model...")
                    best_val_dice_metric = val_dice_metric
                    torch.save(self.model.state_dict(), os.path.join(os.path.join(model_dir, "best.pth")))
                    patience = 0
                else: 
                    print(f"Validation Dice Metric improved slightly from {best_val_dice_metric:.4f} to {val_dice_metric:.4f}, but not significantly enough to reset patience.")
                    torch.save(self.model.state_dict(), os.path.join(os.path.join(model_dir, "best.pth")))
                    if epoch+1 >= self.early_stopping_start: 
                        patience+=1
            else:
                if epoch+1 >= self.early_stopping_start: 
                    patience+=1
            
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(os.path.join(dest_dir, "history.csv"), index=False)

            print("-"*30)
            print(f"This is Best Val Dice Score:  {best_val_dice_metric}")
            print(f"This is Patience {patience}")
            print(f"Training Speed per EPOCH (in seconds): {train_end_time - train_start_time:.4f}")
            print(f"Validation Speed per EPOCH (in seconds): {val_end_time - val_start_time:.4f}")
            
            print(f"Maximum Gigabytes of VRAM Used: {torch.cuda.max_memory_reserved(self.device) * 1e-9:.4f}")
            
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            
            print(f"Train DICE Score EPOCH {epoch+1}: {train_dice_metric:.4f}")
            print(f"Valid DICE Score EPOCH {epoch+1}: {val_dice_metric:.4f}")
            
            print(f"Valid HD95 Score EPOCH {epoch+1}: {val_hd95_metric:.4f}")
            
            print(f"Valid PRECISION Score EPOCH {epoch+1}: {val_precision:.4f}")
            print(f"Valid RECALL Score EPOCH {epoch+1}: {val_recall:.4f}")

            print("-"*30)

            if patience >= self.patience: 
                print(f"\nEARLY STOPPING: Valid Loss did not improve since epoch {epoch+1-patience} with Validation Dice Metric {best_val_dice_metric}, terminating training...")
                break

        torch.save(self.model.state_dict(), os.path.join(os.path.join(model_dir, "last.pth")))
        self.plot_loss_curves(save_path=dest_dir)

def count_parameters(model: torch.nn.Module, only_trainable: bool = True) -> Union[int, List[int]]:
    """
    Counts the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model instance (e.g., a loaded model or a custom nn.Module).
        only_trainable (bool): If True, counts only parameters that require gradients (trainable).
                               If False, returns a list: [trainable_params, total_params].

    Returns:
        Union[int, List[int]]: The total count of parameters (or a list if only_trainable is False).
    """

    # Generator expression to count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if only_trainable:
        return trainable_params
    else:
        # Generator expression to count ALL parameters (trainable + fixed)
        all_params = sum(p.numel() for p in model.parameters())
        return [trainable_params, all_params]

def modify_YOLO(model): 
    old_conv_module = model.model.model.model[0]
    print(old_conv_module)

    # 3. Get the inner nn.Conv2d layer and its weights
    old_nn_conv = old_conv_module.conv
    old_conv_weights = old_nn_conv.weight.data 

    # --- Setup for the new Conv module ---

    # 5. Determine the new nn.Conv2d layer parameters
    # c2 (output channels) comes from the old nn.Conv2d layer
    out_channels = old_nn_conv.out_channels
    # Other parameters from the old nn.Conv2d layer
    kernel_size = old_nn_conv.kernel_size
    stride = old_nn_conv.stride
    padding = old_nn_conv.padding
    dilation = old_nn_conv.dilation
    groups = old_nn_conv.groups
    # Note: YOLO's Conv sets bias=False, so we skip transferring it.

    # 6. Create the new 4-channel nn.Conv2d layer
    new_nn_conv = nn.Conv2d(
        in_channels=4,  # Change from 3 to 4
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=False # Must match the original YOLO structure
    )

    # 7. Initialize the new Conv weights tensor
    # Shape: [out_channels, in_channels (4), kernel_height, kernel_width]
    new_conv_weights = new_nn_conv.weight.data

    # 8. Transfer the original 3-channel weights (RGB)
    new_conv_weights[:, 0:3, :, :] = old_conv_weights[:, 0:3, :, :]

    # 9. Calculate and set the average for the 4th channel
    # Calculate the mean across the input channel dimension (dim=1) of the old weights
    avg_weights = old_conv_weights.mean(dim=1, keepdim=True) 
    # Copy the averaged 3-channel weights to the 4th channel (index 3)
    new_conv_weights[:, 3:4, :, :] = avg_weights

    # 10. Reconstruct the entire Conv module (and keep the original BN and ACT)
    # We can't easily instantiate the original 'Conv' class without the source file 
    # and 'autopad' function, so we modify the existing module's components.

    # Set the new nn.Conv2d layer into the existing Conv module
    old_conv_module.conv = new_nn_conv

    # The BN and ACT layers remain unchanged, which is correct because they operate 
    # on the 'out_channels', which hasn't changed (it's still 16 in your example).
    # You must ensure the BN's num_features matches the Conv's out_channels.

    print("✅ Model's first 'Conv' module successfully modified.")
    print(f"   - New input channels: 4")
    print(f"   - Original output channels: {out_channels}")
    
if __name__ == "__main__": 
    # Create trainer and predictor instances
    p_args = dict(model="yolo_checkpoint/weights/best.pt",
                data=f"data/data.yaml", 
                verbose=True,
                imgsz=160, 
                save=False)

    # Create predictor and Load checkpoint
    YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
    YOLO_predictor.setup_model(p_args["model"])
    modify_YOLO(YOLO_predictor)

    # Create YOLOU instance
    model = YOLOSegPlusPlus(predictor=YOLO_predictor)

    YOLO_predictor.model.to('cpu')

    trainable_count = count_parameters(model, only_trainable=True)
    all_counts = count_parameters(model, only_trainable=False)

    print(f"Total Trainable Parameters: {trainable_count:,}")
    print(f"Total All Parameters (Trainable + Fixed): {all_counts[1]:,}")
    print("-" * 30)
            
    trainer = Trainer(model=model, 
                    data_path="data/stacked_segmentation", 
                    model_path=None, 
                    load_and_train=True,
                    mixed_precision = True,
                
                    epochs=75,
                    image_size = 160,
                    batch_size = 128,
                    lr = 1e-4,

                    early_stopping = True,
                    early_stopping_start = 50,
                    patience = 10, 
                    device = "cuda"
                    )
    trainer.train()
