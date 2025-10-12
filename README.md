# YOLOU: Efficient Segmentation Refinement Module for pre-trained YOLOv12-Seg
Using a pretrained YOLOv12-seg model, YOLOU creates a UNet-like segmentation refinement model, that follows the approach of rough masks (YOLOv12-seg) -> refined masks (YOLOU). The architecture of YOLOU consist of the following: 

	(1) Encoder: YOLOv12-seg pretrained backbone
	(2) Bottleneck: Spatial Transformer Module --> CSPNet modules --> YOLOv12-seg Masks Concat --> ECA Attention Module
	(3) Decoder: Similar to YOLO12-seg pretrained backbone, but Conv2DTranspose + Skip connections

YOLOU performs two passes: 

	(1) First pass, performs inference with YOLOv12-seg, and caches the rough masks, encoder features and downsampled features (skip connections). 
	(2) Second pass, perform inference through YOLOU (starting at bottleneck). 

This caching mechanism, allows for more savings on "compute" on both inference and training. For inference, we only perform a single YOLOv12-seg pass which provides the necessary features for YOLOU inference. For training, we can focus on only training the bottleneck and decoder since the encoder features are fully converged after pretraining. 
The goal of this architecture, it's to create an efficient Brain Tumor Segmentation model for the BraTS-SSA (in low resource and compute environment), that also achieves competitive DICE and Hausffdorf score.

## 📁 Structure
```
├── custom_yolo_predictor/ 	# Ultralytics YOLO Custom Predictor (for 4-channel images)
├── custom_yolo_trainer/	# Ulatralytics YOLO Custom Trainer (for 4-channel images)
├── data/  					# YOLOU dataset
├── modules/    			# YOLOU modules
├── yolo_checkpoint/ 		# Pre-trained YOLOv12-Seg model
├── dataset.py 				# Creates dataset for YOLOU
├── loss.py 				# YOLOU loss
├── predictor.py			# YOLOU predictor
├── trainer.py				# YOLOU trainer
```

## WORK IN PROGRESS
