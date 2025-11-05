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
                act=False)
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
        out = self.eca(out)
        out = out + residual
        return self.activation(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, k1=3, k2=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=k2, padding=1),
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

class SingleLightConv(nn.Module):
    def __init__(self, in_channels, out_channels, k1=3):
        super().__init__()
        self.conv = LightConv(
                c1=in_channels, 
                c2=out_channels, 
                k=k1, 
                act=True)
                
        # 1x1 conv to match channels if needed
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )    

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv(x)
        out+=residual
        return out

class DoubleLightConv(nn.Module):
    def __init__(self, in_channels, out_channels, k1=3, k2=3):
        super().__init__()
        self.conv = nn.Sequential(
            LightConv(
                c1=in_channels, 
                c2=out_channels, 
                k=k1, 
                act=True), 
            LightConv(
                c1=out_channels, 
                c2=out_channels, 
                k=k2, 
                act=True))
                
        # 1x1 conv to match channels if needed
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )    

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv(x)
        out+=residual
        return out
            
