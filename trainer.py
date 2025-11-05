from modules.YOLOUSegPlusPlus import YOLOUSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from dataset import CustomDataset

import os
import time
from typing import Tuple, List, Union
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

from torchvision.transforms.functional import gaussian_blur
import torchvision

class Trainer: 
    def __init__(self,
                model: YOLOUSegPlusPlus,
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

        self.metric = DiceMetric(
            include_background = False, 
            reduction="mean_batch",     
            get_not_nans = False, 
            ignore_empty = False, 
            num_classes = 2,            # 2 stands for [0, 1], technically single class
            return_with_label = False
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
                            # objectmap_sizes = [20, 10])
                            objectmap_sizes = [20])
                            
        val_dataset = CustomDataset(
                            root_path = data_path, 
                            image_path = "images/val", 
                            objectmap_path = "objectmap/val",
                            # heatmap_path = "heatmap/val", 
                            mask_path = "masks/val", 
                            image_size = self.image_size, 
                            # heatmap_sizes = [20])
                            # objectmap_sizes = [20, 10]) # MODIFIED TO INCLUDE P2
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

        ### COMBINING BOTH (NO FREEZING) BUT FORCING ENCODER TO PARAM
        # encoder_params = []
        # for layer in self.model.encoder:
        #     encoder_params += list(layer.parameters())
        # optimizer = torch.optim.Adam(encoder_params + list(self.model.bottleneck.parameters()) + list(self.model.decoder.parameters()), lr=1e-3)

        # for name in trainable_name:
        #     print(f"{name}")

        ### ORIGINAL NO FREEZING DONE
        optimizer = optim.AdamW(trainable_param, lr=self.lr)

        ### ORIGINAL CONFIG WITH FREEZE, ONLY ADD GRADS TO BOTTLENECK AND DECODER, NOT ENCODER
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, self.model.parameters()),
        #     lr=1e-4
        # )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = GradScaler(self.device) # --> mixed precision

        # Initialize variables for callbacks
        self.history = dict(train_loss=[], val_loss=[], train_dice_metric=[], val_dice_metric=[])
        best_val_metric = float("-inf")

        # Create result directory
        dest_dir = f"runs/{self.get_current_time()}" 
        model_dir = os.path.join(dest_dir, "weights")
        self.create_dir(model_dir)

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
            self.metric.reset()
        
            start_time = time.time()
            train_running_loss = 0
            train_running_dice_metric = 0

            if self.mixed_precision:
                for idx, img_mask_heatmap in enumerate(tqdm(train_dataloader)):
                    img = img_mask_heatmap[0].float().to(self.device)
                    mask = img_mask_heatmap[1].float().to(self.device)
                    heatmaps = img_mask_heatmap[2][0].float().to(self.device)
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
                    
                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary  = (pred_sigmoid > 0.5).float()
                    self.metric(pred_binary, mask)

            else:
                for idx, img_mask_heatmap in enumerate(tqdm(train_dataloader)):
                    img = img_mask_heatmap[0].float().to(self.device)
                    mask = img_mask_heatmap[1].float().to(self.device)
                    heatmaps = img_mask_heatmap[2][0].float().to(self.device)
                 
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
                    self.metric(pred_binary, mask)
    
            end_time = time.time()
            train_loss = train_running_loss / (idx + 1)
            train_dice_metric = self.metric.aggregate().item()
            self.metric.reset() # <- Reset again

            self.model.eval()
            val_running_loss = 0
            with torch.no_grad():
                for idx, img_mask_heatmap in enumerate(tqdm(val_dataloader)):
                    img = img_mask_heatmap[0].float().to(self.device)
                    mask = img_mask_heatmap[1].float().to(self.device)
                    
                    heatmaps = img_mask_heatmap[2][0].float().to(self.device)
                    
                    pred = self.model(img, heatmaps)
                    loss = self.loss(pred, mask)

                    # Accumulate Loss and Metrics
                    val_running_loss += loss.item()
                    pred_sigmoid = torch.nn.functional.sigmoid(pred)
                    pred_binary  = (pred_sigmoid > 0.5).float()
                    self.metric(pred_binary, mask)             

                val_loss = val_running_loss / (idx + 1)
                val_dice_metric = self.metric.aggregate().item()
            
            # Update the scheduler
            scheduler.step()
            
            # update the history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice_metric"].append(val_dice_metric)
            self.history["train_dice_metric"].append(train_dice_metric)

            if val_dice_metric > best_val_metric: 
                if abs(best_val_metric - val_dice_metric) > 1e-3:
                    print(f"Validation Dice Metric improved from {best_val_metric:.4f} to {val_dice_metric:.4f}. Saving model...")
                    best_val_metric = val_dice_metric
                    torch.save(self.model.state_dict(), os.path.join(os.path.join(model_dir, "best.pth")))
                    patience = 0
                else: 
                    print(f"Validation Dice Metric improved slightly from {best_val_metric:.4f} to {val_dice_metric:.4f}, but not significantly enough to save the model.")
                    if epoch+1 >= self.early_stopping_start: 
                        patience+=1
            else:
                if epoch+1 >= self.early_stopping_start: 
                    patience+=1
            
            history_df = pd.DataFrame(self.history)
            history_df.to_csv(os.path.join(dest_dir, "history.csv"), index=False)

            print("-"*30)
            print(f"This is Patience {patience}")
            print(f"This is Best Dice Score:  {best_val_metric}")
            print(f"Training Speed per EPOCH (in seconds): {end_time - start_time:.4f}")
            print(f"Maximum Gigabytes of VRAM Used: {torch.cuda.max_memory_reserved(self.device) * 1e-9:.4f}")
            print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
            print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
            print(f"Train DICE Score EPOCH {epoch+1}: {train_dice_metric:.4f}")
            print(f"Valid DICE Score EPOCH {epoch+1}: {val_dice_metric:.4f}")
            print("-"*30)

            if patience >= self.patience: 
                print(f"\nEARLY STOPPING: Valid Loss did not improve since epoch {epoch+1-patience} with Validation Dice Metric {best_val_metric}, terminating training...")
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

    # Create YOLOU instance
    model = YOLOUSegPlusPlus(predictor=YOLO_predictor)

    YOLO_predictor.model.to('cpu')

#     model_dir = "/home/jun/Desktop/inspirit/YOLOU-SegPlusPlus/runs/2025_11_04_11_14_07/weights/best.pth"
# 
#     # Load the checkpoint
#     checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))  # or 'cuda' if using GPU
#     
#     # If checkpoint is a state_dict directly:
#     if 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#     else:
#         state_dict = checkpoint
#     
#     # Load weights into the model
#     model.load_state_dict(state_dict)

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
