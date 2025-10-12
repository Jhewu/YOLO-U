from custom_yolo_predictor.custom_load_images_and_videos import CustomLoadImagesAndVideos
from custom_yolo_predictor.custom_load_pil_and_numpy import CustomLoadPilAndNumpy
from ultralytics.data.build import check_source
from ultralytics.data.loaders import (
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
)

def load_inference_source(source=None, batch: int = 1, vid_stride: int = 1, buffer: bool = False, channels: int = 3):
    """
    Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | Path | torch.Tensor | PIL.Image | np.ndarray, optional): The input source for inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int, optional): The number of input channels for the model.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.

    Examples:
        Load an image source for inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        Load a video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """ 
    
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = CustomLoadPilAndNumpy(source, channels=channels)
    else:
        dataset = CustomLoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
