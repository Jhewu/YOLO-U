from custom_yolo_predictor.custom_load_inference_source import load_inference_source
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import LOGGER

STREAM_WARNING = """
inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""

class CustomBasePredictor(BasePredictor): 
    def setup_source(self, source): 
        """
        Set up source and inference mode.

        Args:
            source (str | Path | List[str] | List[Path] | List[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """

        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "ch", 4),
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}