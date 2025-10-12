from torch import nn
import torch

class ECA(nn.Module):
    def __init__(self, k_size: int = 3):
        super(ECA, self).__init__()
        """
        Constructs a ECA module. Efficient Channel Attention for Conv

        Args: 
            k_size (int): kernel size for Conv1d
        """
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args: 
            x (torch.tensor): input tensor (after mask + x concat)
        Returns:
            (torch.tensor)  : output tensor (same dimensions as input tensor but with attention applied)
        """
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = torch.sigmoid(y)

        return x * y.expand_as(x)
