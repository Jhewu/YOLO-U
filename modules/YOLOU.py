from torch.nn import Sequential, Module
from torch import nn
import torch

from collections import OrderedDict
from typing import List

from ultralytics.nn.modules import (
    ## For the bottleneck
    BottleneckCSP, 

    ## For the decoder reconstruct
    Conv, C3k2, A2C2f, 
    
    ## To upsample in decoder
    ConvTranspose)

from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer
from modules.eca import ECA
from modules.stn import SpatialTransformer

class YOLOU(Module): 
    def __init__(self, 
                 trainer: CustomSegmentationTrainer, 
                 predictor: CustomSegmentationPredictor,
                 target_modules_indices: List[int] = [0, 1, 3, 5, 7]): 
        """
        Creates a YOLOU Network with a Pretrain YOLOv12-Seg model
        Main Idea: rough masks (YOLOv12-Seg) --> fine masks (YOLOU)
        
        Args: 
            trainer (CustomSegmentationTrainer): Custom YOLO segmentation trainer allowing 4-channels
            predictor (CustomSegmentationTrainer): Custom YOLO segmentation predictor allowing 4-channels
            target_modules_indices (list [int]): list of indices to add skip connections (in YOLOv12-Seg every downsample)

        Attributes: 
            yolo_trainer (CustomSegmentationTrainer): Custom YOLO segmentation trainer instance
            yolo_predictor (CustomSegmentationTrainer): Custom YOLO segmentation predictor instance  
            _yolo_seq (nn.Sequential): YOLOv12-Seg model backbone sequence
            encoder (nn.Sequential): Encoder portion extracted from YOLOv12-Seg backbone (first 9 layers)
            decoder (nn.Sequential): Decoder portion constructed from encoder modules
            bottleneck1 (nn.Sequential): First bottleneck layer with BottleneckCSP block
            bottleneck2 (nn.Sequential): Second bottleneck layer with BottleneckCSP block
            stn (SpatialTransformer or None): Spatial Transformer Network for affine transformation
            eca (ECA or None): ECA attention module for feature enhancement
            last_conv (ConvTranspose): Final transposed convolution for output reconstruction
            skip_encoder_indices (set): Set of encoder indices for skip connection targeting
            skip_decoder_indices (set): Set of decoder indices corresponding to skip connections
            skip_connections (list): Cache for storing skip connection features during forward pass
            activation_cache (list): Cache for storing intermediate activations during forward pass

        Methods: 
            _hook_fn: Forward hook function for caching activations (mainly used for YOLOv12-Seg forward pass)
            _assign_hooks: Registers forward hooks on specified modules
            _create_concat_block: Creates concatenation blocks for skip connections
            YOLO_forward: Performs YOLOv12-Seg forward pass to generate initial masks
            _STN_forward: Applies spatial transformer network for affine transformation
            _concat_masks_forward: Concatenates masks with feature maps using ECA attention
            forward: Main forward pass implementation
            _reverse_module_channels: Converts encoder modules to decoder-compatible modules
            _construct_decoder_from_encoder: Builds decoder from encoder modules
            check_encoder_decoder_symmetry: Utility method to verify encoder-decoder symmetry
            print_yolo_named_modules: Debug utility to print all YOLO modules

        Methods: 
        
        TODO: Might reduce target module indices for efficiency
        """

        super().__init__()
        ### YOLOv12-Seg Sections
        self.yolo_trainer = trainer
        self.yolo_predictor = predictor
        self._yolo_seq = self.yolo_predictor.model.model.model

        ### YOLOU Sections
        self.encoder = self._yolo_seq[0:9]
        self.decoder = self._construct_decoder_from_encoder()
        self.bottleneck1 = Sequential(
            BottleneckCSP(c1=256, 
                          c2=256, 
                          n=1, 
                          shortcut=True, 
	                          g=1,
                          e=0.5)).to("cuda")
        self.bottleneck2 = Sequential(
            BottleneckCSP(c1=512, 
                          c2=256, 
                          n=1, 
                          shortcut=True, 
	                          g=1,
                          e=0.5)).to("cuda")
        self.stn = None
        self.eca = None
        # TODO: Make this dynamic and not hardcoded
        self.last_conv = ConvTranspose(
                        c1=16, 
                        c2=1,
                        s=2,
                        p=1, 
                        bn=True,
        
                        # To reconstruct the image size of the respective Conv layer
                        k=4)
        
        ### YOLOU Helper
        self.skip_encoder_indices = set(target_modules_indices)
        self.skip_decoder_indices = set([abs(target_modules_indices[-1] - item + 1) # ---> Reverses and normalizes the indices
                                             for item in target_modules_indices[::-1]])

        self.skip_connections = []
        self.activation_cache = []

        self._assign_hooks()
        self.activation_cache.clear()

    def _hook_fn(self, module, input, output):
        """
        Forward hook, once activate appends the output to
        self.activation_cache
        """
        self.activation_cache.append(output)
        # print(f"\nSuccessfully cached the output {module}\n")
        # print(f"\nSuccessfully cached the output")

    def _assign_hooks(self, modules: list[str] = ["0", 
                                                "1",
                                                "3", 
                                                "5", 
                                                "7",
                                                "8"]):
        """
        Assigns forward hooks for YOLOv12-Seg forward
        Depends on self._hook_fn()

        Args:
            modules (list[str]): List containing the names of the modules
        """        
        found = []
        for name, module in self._yolo_seq.named_modules():
            if name in modules:
                module.register_forward_hook(self._hook_fn)
                print(f"Hook registered on: {name} -> {module}")
                found.append(name)
        
        if not found:
            raise ValueError(f"Modules not found in YOLO")

    def _create_concat_block(self, skip: torch.tensor, x: torch.tensor) -> torch.tensor:
        """
        Creates encoder-to-decoder concat blocks during the decoder step in forward()

        Args:
            skip (torch.tensor): Encoder skip tensor
            x    (torch.tensor): Decoder x tensor from bottleneck

        Returns:
            (torch.tensor): concated tensor of size [B, skip_C + x_C, H, W]
        """
        conv = Conv(
            c1=skip.size(1)+x.size(1), 
            c2=skip.size(1), 
            k=1, 
            s=1).to(x.device)
        return conv(torch.cat([skip, x], dim=1))

    def YOLO_forward(self, x: torch.tensor) -> torch.tensor: 
        """
        YOLOv12-Seg Forward() used at first step of YOLOU

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            (torch.tensor): YOLOv12-Seg masks in batches
            (torch.tensor): cached backbone output (defined in _assign_hook)
        """
        results = self.yolo_predictor(x)

        # Sums the masks and stack it
        mask_batch = []
        for result in results: 
            if result.masks is not None: 
                mask_sum = torch.sum(result.masks.data, dim = 0)
            else: 
                mask_sum = torch.zeros(result.orig_shape[0], result.orig_shape[1]).to("cuda")
            mask_batch.append(mask_sum.unsqueeze(0))

        return torch.stack(mask_batch), self.activation_cache.pop()
    
    def _STN_forward(self, x: torch.tensor) -> torch.tensor: 
        """
        Adaptive STN Forward()
        STN Module Learns Affine transform parameters
        Automatically Set Channel

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            (torch.tensor): Spatially transformed tensor
        """
        self.stn = SpatialTransformer(in_channels=x.size()[1]).to("cuda")
        return self.stn(x)
    
    def _concat_masks_forward(self, masks: torch.tensor, x: torch.tensor) -> torch.tensor: 
        """
        Adapt YOLO predicted, and concantenate
        with CSPNet Bottleneck features while 
        applying ECA-Net attention

        Args:
            masks (torch.tensor): Masks tensor (B,1,160,160)
            x     (torch.tensor): Input tensor

        Returns:
            (torch.tensor): 
        """
        mask = torch.nn.functional.interpolate(masks, size=(5, 5), mode='bilinear').to("cuda")  # -> [B, 1, 5, 5]
        proj_conv = nn.Conv2d(1, 256, kernel_size=1).to("cuda")                                 
        masks = proj_conv(mask)                                                                 # -> [B, 256, 5, 5]
        concat = torch.cat([masks, x], dim=1).to("cuda")                                        # -> [B, 512, 5, 5]
        self.eca = ECA().to("cuda")                                                             # Applying masks attention
        return self.eca(concat)

    def forward(self, x: torch.tensor) -> torch.tensor: 
        """
        Main forward step of YOLOU 

        Args:
            x (torch.tensor): Input tensor [B, 4, H, W]

        Returns:
            x (torch.tensor): Output tensor [B, 4, H, W]
        """
        # encoder
        yolo_x, backbone_x = self.YOLO_forward(x)
        
        # bottleneck
        backbone_x, theta = self._STN_forward(backbone_x.clone())
        x = self.bottleneck1(backbone_x)
        x = self._concat_masks_forward(yolo_x, x)
        x = self.bottleneck2(x)

        # print(f"\nThis is initial activation cache {len(self.activation_cache)}")

        # decoder
        # COMMENT: for some reason self.yolo_predictor(x) triggers forward hooks twice, therefore
        #          take the middle and +1, because we popped at YOLO_forward()
        if len(self.activation_cache) > 6:
            # print("\nActivation cache is greater than 6!")
            self.skip_connections = self.activation_cache[(len(self.activation_cache)//2)+1:]
        else: 
            # print("\nActivation cache is less than 6!")
            self.skip_connections = self.activation_cache

        # Convert .children() -> generator to list 
        children = list(self.decoder.children())
        children = children[:-1]

        for i, layer in enumerate(children):
            layer.to("cuda")

            if i in self.skip_decoder_indices: 
                print(len(self.skip_connections))
                a = self.skip_connections.pop()
                x = self._create_concat_block(a, x)

            x = layer(x)

        # Last layer of the Decoder
        x = self.last_conv(x)

        self.activation_cache.clear()
        self.skip_connections.clear()
        return yolo_x, x

    def _reverse_module_channels(self, module: nn.Module) -> nn.Module:
        """
        In essence, it reverses the input and output of a nn.Module
        present in YOLOv12 backbone. If it's a Conv module then it uses
        a respective ConvTranspose module. It's used to create the
        decoder of YOLOU from the YOLOv12 backbone

        Args:
            module (nn.Module): Module from YOLOv12 Seg backbone

        Returns:
            (nn.Module): Corresponding module to built the decoder
        """

        if isinstance(module, Conv):
            return ConvTranspose(
                c1=module.conv.out_channels, 
                c2=module.conv.in_channels,
                s=module.conv.stride,
                p=module.conv.padding, 
                bn=True,

                # To reconstruct the image size of the respective Conv layer
                k=4)
            
        elif isinstance(module, A2C2f):
            return A2C2f(
                c1=module.cv2.conv.out_channels,
                c2=module.cv1.conv.in_channels,
                 
                # Referencing YOLO12-seg model summary at training start
                a2=True)

        elif isinstance(module, C3k2): 
            return C3k2(
                c1=module.cv2.conv.out_channels, 
                c2=module.cv1.conv.in_channels,
                
                # Referencing YOLO12-seg model summary at training start
                n=2, 
                c3k=False, 
                e=0.25)
    
    def _construct_decoder_from_encoder(self) -> nn.Sequential:
        """
        Construct YOLOU decoder from YOLOv12 Seg backbone. 
        Depends on self._reverse_module_channels()

        Returns:
            (nn.Sequential): Respective decoder sequential list
        """
        decoder_modules = OrderedDict()        
        
        # Iterate through the encoder layers in reverse order
        for name, module in reversed(list(self.encoder.named_children())):
            reversed_module = self._reverse_module_channels(module)
            decoder_modules[f'decoder_{name}'] = reversed_module
                
        return nn.Sequential(decoder_modules)
    
    def check_encoder_decoder_symmetry(self, backbone_last_index: int) -> None: 
        """
        Prints encoder and decoder to check for symmetry

        Args:
            backbone_last_index (int): Last index of the YOLO backbone in YOLOv12 Seg
        """
        for i in range(backbone_last_index): 
            print(f"\n### Comparison {i}:\n{self.encoder[i]}\n")
            print(f"{self.decoder[backbone_last_index - 1 - i]}\n\n")

    def print_yolo_named_modules(self) -> None: 
        """
        Prints YOLOv12-Seg named modules. 
        Used for caching "x" in between modules
        """
        for name, module in self._yolo_seq.named_modules(): 
            print(f"\n{name}")
