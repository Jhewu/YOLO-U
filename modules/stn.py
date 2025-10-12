from torch import nn
import torch

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int, mode: str = 'bilinear'):
        """
        Spatial Transformer Module that predicts affine transformation 
        from backbone features ONLY (no mask concatenation).
        Can be applied to features or input images.

        Args: 
            in_channels (int): input channels
            mode        (str): interpolation method
        """
        super().__init__()
        self.mode = mode

        # Localization network: predicts 6 parameters for 2x3 affine matrix
        self.loc_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, kernel_size=1)  # Output: [B, 6, 1, 1]
        )

        # Initialize to identity transform
        self.loc_net[-1].weight.data.zero_()
        self.loc_net[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args: 
            x (torch.tensor): input tensor [B, C, H, W] — could be image or feature map
        Returns:
            transformed (torch.tensor): warped version of x
            theta       (torch.tensor): affine matrix [B, 2, 3]
        """
        # Predict affine parameters from global pooling
        theta_flat = self.loc_net(x).squeeze(-1).squeeze(-1)  # [B, 6]
        theta = theta_flat.view(-1, 2, 3)  # [B, 2, 3]

        # Create sampling grid
        grid = torch.nn.functional.affine_grid(theta, x.size(), align_corners=False)

        # Sample input
        transformed = torch.nn.functional.grid_sample(
            x, grid,
            mode=self.mode,
            padding_mode='border',
            align_corners=False
        )

        return transformed, theta
