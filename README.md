# YOLOU-Seg++: an Improved 1.5 Stage YOLO Segmentation Model (for BraTS-SSA) in PyTorch 
**YOLOU-Seg++** is an improved "1.5 stage" YOLO segmentation model that leverages YOLO's strenghts in classification and location strength, and UNet-like strength in pixel-level segmentation, with a smart tensor caching mechanism, that improves over vanilla YOLO segmentation, and serves as an "add-in" module to convert a strong YOLO detection model into a segmentation model. Whereas YOLO segmentation models are often constraint by the bbox produced by the detect branch (for efficiency reasons), YOLOU-Seg++ fuses the bounding box at skip connections as a gating/suggestive mechanism, allowing the model full context of the image. 

[INSERT YOLOU-SEG++ DIAGRAM]

YOLOU-Seg++ performs 1.5 passes: 

	(1) **1st Pass**: forward pass with  YOLOv12, caching  the bbox coordinates, backbone output tensors and downsampled tensors (for skip connections). 
	(2) **0.5 Pass**: forward pass (with the backbone output tensors) to the CSP bottleneck, and then decoder, where skip connections concatenates with backbone downsampled tensors and with the bbox coordinates (spatial guidance)

This caching mechanism, allows for more savings on "compute" on both inference and training. For inference, we only perform a single YOLOv12 forward pass, providing the necessary tensors for YOLOU-Seg++ inference (YOLOv12 backbone act as the encoder to our UNet like architecture). For training, we will only be training the YOLOU-Seg++ bottleneck and decoder (since the backbone is already converged). 

The goal of this architecture, it's to create an efficient Brain Tumor Segmentation model for the BraTS-SSA (in low resource and compute environment), that also achieves competitive DICE and Hausffdorf score, and to serve as a "add-in" module for a strong YOLO detection model, where the researcher wishes to transform it into a segmentation model, with minimal training. 

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
