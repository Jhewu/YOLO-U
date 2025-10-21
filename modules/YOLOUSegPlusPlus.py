from torch.nn import Sequential, Module, Upsample, Sigmoid
from torch import nn
import torch

from collections import OrderedDict
from typing import List

from ultralytics.nn.modules import (
    ## For the bottleneck
    BottleneckCSP, 

    ## For the decoder reconstruct
    Conv, C3k2, A2C2f)

# local/custom scripts
from custom_yolo_predictor.custom_detseg_predictor import CustomDetectionPredictor
from custom_yolo_trainer.custom_trainer import CustomDetectionTrainer
from modules.eca import ECA
from modules.stn import SpatialTransformer

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
        ### YOLO12 (detect) Section
        self.yolo_predictor = predictor # for inference

        ### YOLOU-Seg++ Modules Section
        self.encoder = predictor.model.model.model[0:9] # <- Obtain sequential modules from 0-8 for training
        self.bottleneck = Sequential(        
                SpatialTransformer(in_channels=self.encoder[-1].cv2.conv.out_channels), # <- Learns geometric transformation, to correct encoder features
                ECA(), # <- weights the channels
                BottleneckCSP(c1=256, 
                              c2=256, 
                              n=2, 
                              shortcut=True, 
                              g=1,
                              e=0.5))
        self.decoder = self._construct_decoder_from_encoder()[:-1] # <- Skip the last conv layer, to output binary masks
        self.last_conv = Conv(
                c1=16, 
                c2=1,
                k=3
                )
                
        ### YOLOU-Seg++ Helper (Separate Modules Used in forward() in Decoder)
        self._upsample = Upsample(scale_factor = 2, mode = "bilinear", align_corners = True)
        self._sigmoid = Sigmoid()
        self._ecas = nn.ModuleList( [ECA() for i in range( len(target_modules_indices ) )] )
        self._heatmap_params = nn.ParameterList( [nn.Parameter(torch.tensor(1.0)) for i in range( len(target_modules_indices) - 1)] )
        self._heatmap_proj = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1,
                out_channels = 128, 
                kernel_size = 1) 
                for i in range(2)])
        self._concat_proj = nn.ModuleList([ 
            nn.Conv2d(
                in_channels = 128*2,
                out_channels = 128, 
                kernel_size = 1),   # <- A2C2F (Module 6 in YOLO Backbone)
            nn.Conv2d(
                in_channels = 128*2,
                out_channels = 128, 
                kernel_size = 1),   # <- C3k2 (Module 4 in YOLO Backbone)
            nn.Conv2d(
                in_channels = 64*2,
                out_channels = 64, 
                kernel_size = 1)])  # <- C3k2 (Module 2 in YOLO Backbone)
        self._residual_proj = nn.ModuleList([
            nn.Conv2d(
                in_channels = 64*2,
                out_channels = 64, 
                kernel_size = 1), 
            nn.Conv2d(
                in_channels = 32*2, 
                out_channels = 32, 
                kernel_size = 1),      
        ])
        
        ### Miscellaneous Section
        self.verbose = verbose
        self.skip_connections = []
        self._indices = {
                "upsample": set([1, 3, 5, 7, 8]), 
                "skip_connections_encoder": set(target_modules_indices), 
                "skip_connections_decoder": set([2, 4, 6]), 
                "eca":0, "heatmap_param": 0, "heatmap_proj": 0, "concat_proj": 0, "residual_proj": 0} 
        
        ### TODO: IMPLEMENT THIS DYNAMIC DECODER INDICES LATER
        # self.skip_decoder_indices = set([abs(target_modules_indices[-1] - item + 1) # ---> Reverses and normalizes the indices
                                             # for item in target_modules_indices[::-1]])
        
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
            (torch.tensor): YOLOv12 bbox in batches
            (torch.tensor): cached backbone output (defined in _assign_hook)
        """

        with torch.no_grad():
            results = self.yolo_predictor(x)

        # Sums the boxes and stack it
        box_batch = torch.zeros( (len(results), 1, *results[0].orig_shape), device='cuda') # [B, C, H, W]
        for i, result in enumerate(results):
            if result.boxes:
                box_batch[i] = torch.sum(result.boxes, dim=0).unsqueeze(0)

        return mask_batch, self.activation_cache.pop()
    
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

    def inference(self): 
        """
        TODO: IMPLEMENT INFERENCE WITH FORWARD HOOKS
        """

    
        #         # decoder
        # # COMMENT: for some reason self.yolo_predictor(x) triggers forward hooks twice, therefore
        # #          take the middle and +1, because we popped at YOLO_forward()
        # if len(self.activation_cache) > 6:
        #     # print("\nActivation cache is greater than 6!")
        #     self.skip_connections = self.activation_cache[(len(self.activation_cache)//2)+1:]
        # else: 
        #     # print("\nActivation cache is less than 6!")
        #     self.skip_connections = self.activation_cache


        return
    def _reset_indices(self) -> None: 
        """
        Resets the indices used to keep track of the independent modules found in the Decoder section
        """
        self._indices["eca"]           = 0
        self._indices["heatmap_param"] = 0
        self._indices["heatmap_proj"]  = 0   
        self._indices["concat_proj"]   = 0
        self._indices["residual_proj"] = 0

    def forward(self, x: torch.tensor, heatmaps: List[torch.tensor]) -> torch.tensor: 
        """
        Main Forward Step for YOLOU-Seg++ (for Training)
        Use self.inference() for inference 

        Args:
            x        (torch.tensor):        Input tensor [B, 4, H, W]
            heatmaps (List[torch.tensor]):  List of resized heatmaps tensors to concatenate at skips [1, 1, h, w], where h and w are resized heights and weights (< H and W)

        Returns:
            x (torch.tensor): Output tensor [B, 4, H, W]

        TODO: Implement Residual Connections for the non-YOLO Layers
        """
        # Reset each forward()
        self.skip_connections = []            
        self._reset_indices()

        # Encoder (frozen)
        with torch.no_grad(): 
            for idx, module in enumerate(self.encoder): 
                x = module(x)  
                if idx in self._indices.get("skip_connections_encoder"):
                    self.skip_connections.append(x)                                 # <- Manually cache tensors for skips

        # Bottleneck (trainable)
        x = self.bottleneck(x)    

        # Decoder (trainable)
        for idx, module in enumerate(self.decoder):                                 # <- Remove last Conv, because last Conv must be 1-channel (binary mask)
            if idx in self._indices.get("upsample"): x = self._upsample(x) 
            if idx in self._indices.get("skip_connections_decoder"):            
                skip, size = self.skip_connections.pop(), x.size()[1]
                if heatmaps:                                                        # <- If there are heatmaps, we fuse the heatmap (learnable gating) with skips
                    heatmap = heatmaps.pop()
                    fusion = self._heatmap_proj[ self._indices.get("heatmap_proj") ]  # <- Get fusion module
                    self._indices["heatmap_proj"]+=1                                 # <- Increase idx
                            
                    gate = self._sigmoid(fusion(heatmap))
                    skip = skip * (1.0 + self._heatmap_params[self._indices.get("heatmap_param")] * gate) 
                    self._indices["heatmap_param"]+=1                         # <- Increase idx

                x = torch.cat([skip, x], dim=1)                 # <- Concatenate
                x = self._ecas[ self._indices.get("eca") ](x)   # <- Re-weight the Concat channels
                self._indices["eca"]+=1    

                x = self._concat_proj[ self._indices.get("concat_proj") ](x)
                self._indices["concat_proj"]+=1

            if idx not in self._indices.get("upsample"): # <- If non-upsampling blocks (YOLO Modules), add residual (better gradient flow)
                residual = x.clone() 
                x = module(x)
                if x.shape == residual.shape: 
                    x = x + residual
                else: 
                    proj = self._residual_proj[ self._indices.get("residual_proj") ]
                    self._indices["residual_proj"]+=1
                    x = x + proj(residual)
            else: x = module(x)

        # Output (trainable)
        x = self.last_conv( self._upsample(x) )
        return x

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

        if isinstance(module, Conv):
            return Conv(
                c1=module.conv.out_channels, 
                c2=module.conv.in_channels,
                k=module.conv.kernel_size, 
                p=module.conv.padding)
            
        elif isinstance(module, A2C2f):
            return A2C2f(
                c1=module.cv2.conv.out_channels,
                c2=module.cv1.conv.in_channels,
                
                # Referencing YOLOv12 (detect) model summary (see docstrings)
                area=1,
                n=2,
                a2=True)
        elif isinstance(module, C3k2): 
            return C3k2(
                c1=module.cv2.conv.out_channels, 
                c2=module.cv1.conv.in_channels,
                
                # Referencing YOLOv12 (detect) model summary (see docstrings)
                n=1, 
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
    
    def check_encoder_decoder_symmetry(self, backbone_last_index: int = 9) -> None: 
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
        Prints YOLOv12 (detect) named modules. 
        Used for caching "x" in between modules
        """
        for name, module in self.encoder.named_modules(): 
            print(f"\n{name}")

