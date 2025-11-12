# YOLO-Seg++: Lighter, Higher-Quality YOLO-Based Brain Tumor Segmentation for Low-Resource Environment (BraTS-SSA)

**WORK IN PROGFESS** Current state-of-the-art brain tumor segmentation and semantic segmentation models rely on expensive 3D tensor and self-attention (Transformer) calculations. While a better-performing model can always be found with access to computing, these models cannot be deployed in low-resource settings or edge devices. Current low-parameter segmentation models, such as YOLOv12n-seg, perform great on day-to-day low-risk tasks; however, their segmentation quality suffers greatly and cannot be relied on for high-risk scenarios such as brain tumor segmentation, opening areas for exploration. To address the challenge of efficient and accurate medical image segmentation, we propose YOLO-Seg++, a lightweight, parameter-efficient (2.6M + 80K) segmentation model that outperforms YOLOv12Seg (2.9M) and vanilla UNet (7M) with Dice Scores of 0.81, 0.84, and 0.87 for whole tumors, respectively, on the BraTS-SSA dataset. YOLO-Seg++ leverages the logits from a YOLO detection model, which our analysis shows to encode strong spatial localization cues of tumor regions, as a semantic bottleneck. These logits are integrated with YOLO’s backbone “skip” features to form a compact UNet-like architecture. By combining YOLO’s strength in detection and localization with UNet’s pixel-level precision, our approach enables the conversion of a strong YOLO detector (transfer learning) into an equally strong segmentation model with minimal additional parameters and near-native CPU inference performance. Decoupling the YOLO detector and segmentation head further enhances this adaptability, allowing task-specific loss optimization beyond YOLO’s coupled training objective. Efficient and accurate brain tumor segmentation can be achieved with the YOLO-Seg++ architecture, using pre-existing and well-established research on model architectural design. Further research in data augmentation, customized loss functions, advanced hyperparameter search, and ultimately architectural refinement can improve and suggest potential for future clinical adaptation pending further validation.

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

## REPOSITORY IS WORK IN PROGRESS
