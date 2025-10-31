import torch
import torch.nn as nn

from ultralytics.nn.modules import LightConv, DWConv, GhostConv
from modules.eca import ECA

class GhostBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, g_k = 3, d_k = 5):
        super().__init__()
        self.ghost = GhostConv(
                c1=in_channels, 
                c2=out_channels, 
                k=g_k, 
                act=True)
    
        self.dw = DWConv(
                c1=out_channels, 
                c2=out_channels, 
                k=d_k, 
                act=False)

        self.eca = ECA()
        
        # 1x1 conv to match channels if needed
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )    
        
        self.activation = nn.SiLU(inplace = True)
        
    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.ghost(x)
        out = self.dw(out)
        out = self.eca(out+residual)
        return self.activation(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU(inplace = True)
        
        # 1x1 conv to match channels if needed
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )    

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv(x)
        out+=residual
        return self.activation(out)

class DoubleLightConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            LightConv(
                c1=in_channels, 
                c2=out_channels, 
                k=3, 
                act=True), 
            LightConv(
                c1=out_channels, 
                c2=out_channels, 
                k=3, 
                act=True))
        self.activation = nn.SiLU(inplace = True)
        
        # 1x1 conv to match channels if needed
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )    

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv(x)
        out+=residual
        # return self.activation(out)
        return out
            
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       x1 = self.up(x1)
       x = torch.cat([x1, x2], 1)
       return self.conv(x)


