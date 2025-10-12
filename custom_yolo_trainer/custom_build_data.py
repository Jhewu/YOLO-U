from custom_yolo_trainer.custom_data import CustomYOLODataset
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import colorstr
from typing import Any, Dict

def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: Dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
):
    
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = CustomYOLODataset # YOLOMultiModalDataset if multi_modal else CustomYOLODataset
    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )