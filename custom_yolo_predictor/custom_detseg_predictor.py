from custom_yolo_predictor.custom_predictor import CustomBasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops

from ultralytics.utils.ops import crop_mask, scale_masks
import torch.nn.functional as F

def process_mask(protos, masks_in, bboxes, shape, upsample: bool = False):
    """
    Apply masks to bounding boxes using mask head output.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).
        upsample (bool): Whether to upsample masks to original image size.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # CHW
    width_ratio = mw / iw
    height_ratio = mh / ih

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= width_ratio
    downsampled_bboxes[:, 2] *= width_ratio
    downsampled_bboxes[:, 3] *= height_ratio
    downsampled_bboxes[:, 1] *= height_ratio

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]  # CHW
    # return masks.gt_(0.0)
    return masks

def process_mask_native(protos, masks_in, bboxes, shape):
    """
    Apply masks to bounding boxes using mask head output with native upsampling.

    Args:
        protos (torch.Tensor): Mask prototypes with shape (mask_dim, mask_h, mask_w).
        masks_in (torch.Tensor): Mask coefficients with shape (N, mask_dim) where N is number of masks after NMS.
        bboxes (torch.Tensor): Bounding boxes with shape (N, 4) where N is number of masks after NMS.
        shape (tuple): Input image size as (height, width).

    Returns:
        (torch.Tensor): Binary mask tensor with shape (H, W, N).
    """
    c, mh, mw = protos.shape  # CHW
    masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    masks = scale_masks(masks[None], shape)[0]  # CHW
    masks = crop_mask(masks, bboxes)  # CHW
    # return masks.gt_(0.0)
    return masks

class CustomDetectionPredictor(CustomBasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def get_obj_feats(self, feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min([x.shape[1] for x in feat_maps])  # find smallest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if len(idx) else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.

        Returns:
            (List[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

class CustomSegmentationPredictor(CustomDetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """
        Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch.
                Each Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo11n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # Extract protos - tuple if PyTorch model or array if exported
        protos = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        # print(f"\nThis is protos {protos.size()}")
        return super().postprocess(preds[0], img, orig_imgs, protos=protos)

    def construct_results(self, preds, img, orig_imgs, protos):
        """
        Construct a list of result objects from the predictions.

        Args:
            preds (List[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (List[np.ndarray]): List of original images before preprocessing.
            protos (List[torch.Tensor]): List of prototype masks.

        Returns:
            (List[Results]): List of result objects containing the original images, image paths, class names,
                bounding boxes, and masks.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path, proto)
            for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto):
        """
        Construct a single result object from the prediction.

        Args:
            pred (np.ndarray): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        if not len(pred):  # save empty boxes
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

    # def construct_result(self, pred, img, orig_img, img_path, proto):
    #     """
    #     Construct a single result object from the prediction.

    #     Args:
    #         pred (np.ndarray): The predicted bounding boxes, scores, and masks.
    #         img (torch.Tensor): The image after preprocessing.
    #         orig_img (np.ndarray): The original image before preprocessing.
    #         img_path (str): The path to the original image.
    #         proto (torch.Tensor): The prototype masks.

    #     Returns:
    #         (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
    #     """
    #     if not len(pred):  # save empty boxes
    #         masks = None
    #     elif self.args.retina_masks:
    #         pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    #         masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
    #     else:
    #         masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
    #         pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    #     if masks is not None:
    #         keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
    #         pred, masks = pred[keep], masks[keep]
    #     return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
