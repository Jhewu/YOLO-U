from torch.nn import Sequential, Module, Upsample
from torch import nn
import torch

from typing import List

from ultralytics.nn.modules import C3Ghost

# local/custom scripts
from custom_yolo_predictor.custom_detseg_predictor import CustomDetectionPredictor
from modules.eca import ECA

import torchvision
from torchvision.transforms import GaussianBlur
from modules.unet_parts import DoubleLightConv, DoubleConv, GhostBlock

class YOLOUSegPlusPlus(Module): 
    def __init__(self,
                 predictor: CustomDetectionPredictor,
                 verbose: bool = False,
                 target_modules_indices: List[int] = [2, 4, 6]): 
        """
        WARNING: DOCUMENTATION NOT UPDATED
        
        Creates a YOLOU-Seg++ Network with Pretrained YOLOv12 (detection) model
        Main Idea: Using YOLOv12 bbox as guidance in UNet skip connections and recycling YOLOv12 backbone as the encoder
        
        Args: 
            predictor (CustomSegmentationTrainer): Custom YOLO segmentation predictor allowing 4-channels
            target_modules_indices (list [int]): list of indices to add skip connections (in YOLOv12-Seg every downsample)

        Attributes: 
            yolo_predictor (CustomSegmentationTrainer): Custom YOLO segmentation predictor instance  
            encoder (nn.Sequential): Encoder portion extracted from YOLOv12-Seg backbone (first 9 layers)
            decoder (nn.Sequential): Decoder portion constructed from encoder modules
            bottleneck (nn.Sequential): First bottleneck layer with BottleneckCSP block

        Methods: 
            _hook_fn: Forward hook function for caching activations (mainly used for YOLOv12-Seg forward pass)
            _assign_hooks: Registers forward hooks on specified modules
            _create_concat_block: Creates concatenation blocks for skip connections
            YOLO_forward: Performs YOLOv12-Seg forward pass to generate initial masks
            _STN_forward: Applies spatial transformer network for affine transformation
            forward: Main forward pass implementation
            _reverse_module_channels: Converts encoder modules to decoder-compatible modules
            _construct_decoder_from_encoder: Builds decoder from encoder modules
            check_encoder_decoder_symmetry: Utility method to verify encoder-decoder symmetry
            print_yolo_named_modules: Debug utility to print all YOLO modules

        -------------------------------------------------------------------------------------------------------------
        YOLOv12 backbone
        -------------------------------------------------------------------------------------------------------------
                                   from  n    params  module                                       arguments
          0                  -1  1       608  ultralytics.nn.modules.conv.Conv             [4, 16, 3, 2]
          1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
          2                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]
          3                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
          4                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]
          5                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
          6                  -1  2    180864  ultralytics.nn.modules.block.A2C2f           [128, 128, 2, True, 4]
          7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
          8                  -1  2    689408  ultralytics.nn.modules.block.A2C2f           [256, 256, 2, True, 1]
        -------------------------------------------------------------------------------------------------------------
    
        """
        ### TODO
        ## (1) Make the individual modules declaration adaptive
        ## (2) Use iter() instead of indices, which might be prone to breaking
        ## (3) Update Documentation
        ## (4) Implement Inference Step
        ## (5) Clean up Methods

        super().__init__()
        ### YOLO predictor and backbone
        self.yolo_predictor = predictor.model.model         # <- For inference
        self.encoder = nn.ModuleList(module for module in predictor.model.model.model[0:5]) 
        for param in self.encoder.parameters(): # <- Frozen
            param.requires_grad = False
        self.encoder.eval() 

        ### Lightweight Decoder
        self.input = DoubleLightConv(1, 64) # 20x20 Spatial Resolution
        self.decoder = nn.ModuleList([
            Sequential(
                C3Ghost(64+128, 96),
                DoubleLightConv(96, 96),    # <- Mixing (64 Input) + (128 Skip)    
            ),
            ECA(), 
            DoubleLightConv(96, 64),        # <- Assume Upsample Here 20x20 -> 40x40
            Sequential(
                C3Ghost(64+64, 64),
                DoubleLightConv(64, 64),    # <- Mixing (32 Input) + (64 Skip)    
            ),
            ECA(),    
            DoubleLightConv(64, 32),        # <- Assume Upsample Here 40x40 -> 80x80 + (32 Skip) 
            DoubleLightConv(32, 16),        # <- Assume Upsample Here 80x80 -> 160x160
        ])
        self.output = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1) # <- Assume Upsample Here 80x80 -> 160x160
        
        ### Miscellaneous Section
        self.upsample = Upsample(scale_factor = 2, mode = "bilinear", align_corners = False)
        self.sigmoid = nn.Sigmoid()
        self.resize = torchvision.transforms.Resize((40, 40), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        self.verbose = verbose
        self.skip_connections = []
        self._indices = {
                "upsample": set([2, 5, 6]),                
                "skip_connections_encoder": set([2, 4]), 
                "skip_connections_decoder": set([0, 3])}

                ### FUTURE: ADAPT LATER
                # "upsample": set([1, 3, 5, 7, 8]), 
                # "skip_connections_encoder": set(target_modules_indices), # [2, 4, 6, 8]
                # "skip_connections_decoder": set( [ abs(item-8) for item in reversed(target_modules_indices) ] )} 
        
        if torch.cuda.is_available():
            print(f"\nATTENTION: CUDA {torch.cuda.get_device_name(0)} is available, forwarding YOLOv12 backbone twice is faster than forward hooks...\n")            
        else: 
            print(f"\nATTENTION: CUDA is not available (CPU), using forward hooks to save on compute...\n")
            self.activation_cache = []
            self._assign_hooks()    

    def _hook_fn(self, module, input, output):
        """
        Forward hook, once activate appends the output to
        self.activation_cache
        """
        self.activation_cache.append(output)
        if self.verbose:
            print(f"\nSuccessfully cached the output {module}\n")

    def _assign_hooks(self, modules: list[str] = ["0", 
                                                "1",
                                                "3", 
                                                "5", 
                                                "7",
                                                "8"]):
        """
        TODO: REWORK THIS METHOD
        Assigns forward hooks for YOLOv12-Seg forward
        Depends on self._hook_fn()

        Args:
            modules (list[str]): List containing the names of the modules
        """        
        found = []
        for name, module in self.encoder.named_modules():
            if name in modules:
                module.register_forward_hook(self._hook_fn)
                if verbose: 
                    print(f"Hook registered on: {name} -> {module}")
                found.append(name)
        
        if not found:
            raise ValueError(f"Modules not found in YOLO")
            
    def inference(self): 
        """
        FUTURE: Work in progress
        """
        pass

    def forward(self, x: torch.tensor, logits: torch.tensor) -> torch.tensor: 
        """
        Main Forward Step for YOLOSeg++ (for Training)
        Use self.inference() for inference 

        Args:
            x        (torch.tensor):  Input tensor [B, 4, H, W]
            heatmaps (torch.tensor):  List of resized heatmaps tensors to concatenate at skips [1, 1, h, w], where h and w are resized heights and weights (< H and W)

        Returns:
            x (torch.tensor): Output tensor [B, 4, H, W]
        """
        # Encoder (weights frozen in training loop)   
        self.skip_connections = []
        for idx, module in enumerate(self.encoder):   
            x = module(x)  
            if idx in self._indices.get("skip_connections_encoder"):
                self.skip_connections.append(x) # <- Manually cache tensors for skips

        # Decoder (trainable)
        x = self.input(logits)
        for idx, module in enumerate(self.decoder): 
            if idx in self._indices.get("skip_connections_decoder"): 
                skip = self.skip_connections.pop()
                x = torch.concat([x, skip], dim=1)
            if idx in self._indices.get("upsample"): 
                x = self.upsample(x)
            x = module(x)
        out = self.output(x)
        return out
