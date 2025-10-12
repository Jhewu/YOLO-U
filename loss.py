import torch
from monai.losses import HausdorffDTLoss, TverskyLoss, DiceLoss, FocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

class YOLOULoss(torch.nn.Module): 
    """
    Loss for YOLOU: Dice Loss + Hausdorff Distance + Focal Loss

    Args:
        a (float): Bias for Dice & Tversky Loss (shared equally)
        b (float): Bias for Hausdorff Distance Loss 
        c (float): Bias for Focal Loss (only on pixels that Stage 1 Missed)
    """
    def __init__(self,  a: float = 1.0, 
                        b: float = 0.5,
                        c: float = 0.3,
                        ): 
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=0.6, beta=0.4) # --> Emphasize precision
        self.hausdorff = HausdorffDTLoss(include_background=False)
        self.focal = FocalLoss(reduction="none")
        
    def forward(self, 
                    preds: torch.tensor, 
                    init_preds: torch.tensor,
                    targets: torch.tensor,
                    sigmoid: bool = True ) -> torch.tensor:
        """
        Computes YOLOU composite loss. 

        Args: 
            preds      (torch.tensor): YOLOU predicted tensor
            init_preds (torch.tensor): YOLOv12-Seg predicted tensor (course masks)
            target     (torch.tensor): Ground truth tensor
            sigmoid    (bool)        : If True apply Sigmoid, else no

        Returns: 
            total_loss (torch.tensor)
        """

        if sigmoid: 
            preds = torch.sigmoid(preds)
            init_preds = torch.sigmoid(init_preds)
    
        # Only penalize pixels where GT says tumor exists, but Stage 1 missed it
        fn_mask = (targets == 1) & (init_preds < 0.1)  # [B, 1, H, W]
        focal_loss = (self.focal(preds, targets) * fn_mask.float()).mean()
    
        total_loss = (
            (self.a/2) * self.dice(preds, targets) +
            (self.a/2) * self.tversky(preds, targets) +
            self.b * self.hausdorff(preds, targets) +
            self.c * focal_loss
            )
        return total_loss

def dice_metric(pred, target, smooth=1e-8):
    """
    Computes the Dice Score/Coefficient for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W) - logits or probabilities
        target: Tensor of ground truth (batch_size, 1, H, W) - binary masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Score (higher is better)
    """
    # Apply sigmoid to convert logits to probabilities if needed
    pred = torch.nn.functional.sigmoid(pred)
    
    # Flatten tensors for easier computation (optional but common)
    pred_flat = pred.view(pred.size(0), -1)  # (batch_size, H*W)
    target_flat = target.view(target.size(0), -1)  # (batch_size, H*W)
    
    # Calculate intersection
    intersection = (pred_flat * target_flat).sum(dim=1)  # Per batch
    
    # Calculate Dice coefficient
    # Formula: 2 * |A ∩ B| / (|A| + |B|)
    dice_denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (dice_denominator + smooth)
    
    return dice.mean()  # Average across batch

def dice_loss(pred: torch.tensor, target:torch.tensor, smooth: float = 1e-8):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W) - logits
        target: Tensor of ground truth (batch_size, 1, H, W) - binary masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Loss (lower is better)
    """
    dice_score = dice_metric(pred, target, smooth)
    return 1 - dice_score


