from custom_yolo_predictor.custom_detseg_predictor import CustomDetectionPredictor

import torch
from torchvision.utils import save_image

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import argparse
import os

def create_dir(path:str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def generate_heatmaps(): 
    """
    Generate heatmaps from YOLO Prediction Result object and saves
    it in the destination directory. Refer to the if __name__ == "__main__"
    for information about the global function parameters
    """

    """------START OF HELPER FUNCTIONS------"""
    def add_gaussian_heatmap_to_canvas(canvas: torch.tensor, box_conf: torch.tensor, center_x: int, center_y: int, width: int, height: int, device='cuda') -> torch.tensor:
        """
        Creates a Gaussian heatmap in global canvas space and adds it directly to canvas.
        
        Args: 
            canvas (torch.tensor): canvas tensor, shape (1, 1, canvas_height, canvas_width)
            box_conf (torch.tensor): box confidence, used as multiplier for the strength of the signal
            center_x (int): center x coordinate in global canvas space
            center_y (int): center y coordinate in global canvas space
            width (int):    width of the bounding box (for sigma calculation)
            height (int):   height of the bounding box (for sigma calculation)
            device (str):   device to cast
        
        Returns: 
            canvas (torch.tensor): canvas with Gaussian heatmap added
        """
        _, _, canvas_height, canvas_width = canvas.shape
        
        # Create Gaussian in global canvas coordinate space
        sigma = 0.15 * max(height, width)
        y, x = torch.meshgrid(torch.arange(canvas_height, dtype=torch.float32, device=device), 
                              torch.arange(canvas_width, dtype=torch.float32, device=device), 
                              indexing='ij')
        
        # Calculate Gaussian centered at (center_x, center_y)
        gaussian = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)) * box_conf
        
        # Add directly to canvas
        canvas[0, 0] = canvas[0, 0] + gaussian
        
        return canvas
        
    def generate_heatmaps_from_bbox(results: List, heatmap_dest_dir: str) -> None: 
        """
        Creates the heatmaps from YOLO bounding boxes and saves the heatmap
        
        Args: 
            results (List[Result]): Ultralytics result object, reference: https://docs.ultralytics.com/modes/predict/#working-with-results
            heatmap_dest_dir (str): heatmap destination directory
        
        """
        for result in results: 
            boxes = result.boxes
            path = result.path
            canvas = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device="cuda") # Initial heatmap with zeros
            dest_dir = os.path.join(heatmap_dest_dir, os.path.basename(path))
            
            if boxes:   # <- if there are predictions    
                for box in boxes: # These are individual boxes
                    box_conf = box.conf ; coord = box.xywh[0]
                    center_x,   center_y =    int(coord[0]), int(coord[1])
                    width,      height  =     int(coord[2]), int(coord[3])
                    canvas = add_gaussian_heatmap_to_canvas(canvas, box_conf, center_x, center_y, width, height) # <- box confidence will determine the strength of the signal

                save_image(canvas, dest_dir)
                print(f"SAVING HEATMAP: Prediction in... {path}")
            else:       # <- if there are no predictions
                save_image(canvas, dest_dir)
                print(f"SAVING EMPTY: No prediction in... {path}")
                
    """------END OF HELPER FUNCTIONS------"""
    image_dir           =  os.path.join(IN_DIR,  "images")
    heatmap_dest_dir    =  os.path.join(IN_DIR, "heatmap")

    # Declare args for custom YOLO
    args = dict(save=False, verbose=False, device=DEVICE, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, conf=CONFIDENCE) # ADDED CONFIDENCE THRESHOLD HERE
    predictor = CustomDetectionPredictor(overrides=args)
    predictor.setup_model(YOLO_DIR)

    for split in ["test", "train", "val"]: 
        ## Directories are structured as 'dataset/images/split/image1, image2,..., imagen'
        image_split         = os.path.join(image_dir, split)    
        heatmap_split       = os.path.join(heatmap_dest_dir, split)
        create_dir(heatmap_split)

        ## Create full paths of images. The line below does the following: 
        # (1) Gets the list of images
        # (2) Sort the images (ensure images order)
        # (3) Create the full paths of each image
        image_full_paths = [os.path.join(image_split, image) for image in sorted( os.listdir(image_split) )]

        ### ----------------------------------------
        # Single Batch | Inference Per Images
        # for image_path in image_full_paths[:50]: 
        #     result = predictor(image_path)
        #     generate_heatmaps_from_bbox(result, heatmap_split)

        ### ----------------------------------------
        # Multi Batch | Multiple Inference Per Batch (Multi-Images)

        batches = [image_full_paths[i:i + BATCH_SIZE] for i in range(0, len(image_full_paths), BATCH_SIZE)]

        with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
            futures = []
            for batch in batches: 
                batch_results = predictor(batch)
                for result in batch_results: 
                    # Submit tasks and store the future
                    future = executor.submit(generate_heatmaps_from_bbox, [result], heatmap_split)
                    futures.append(future)
            # Wait for all futures to complete before exiting the with block
            for future in as_completed(futures): 
                try: 
                    future.result() # This will raise any exceptions that occurred in the thread
                except Exception as e: 
                    print(f"Error processing heatmap: {e}")
        
        ### ----------------------------------------

if __name__ == "__main__": 
    # ---------------------------------------------------
    des="""
    Using a pre-trained YOLO Ultralytics model (YOLOv12n) 
    to extract bounding boxes from a BraTS dataset, and generate 
    soft gaussian heatmaps for YOLO-UNet training. Heatmaps
    will be saved as a separate directory (to not modified original
    preprocessed images), and its intensity based on YOLO confidence
    """
    # ---------------------------------------------------

    ### Parse arguments
    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data_dir", type=str,help='directory (root) of BraTS dataset\t[stacked_segmentation]')
    parser.add_argument("--yolo_dir", type=str,help='directory of YOLO Ultralytics model weights\t[yolo_checkpoint/weights/best.pt]')
    parser.add_argument('--out_dir',type=str,help='output directory of the generated heatmaps\t[heatmaps]')
    parser.add_argument("--device", type=str,help='cpu or cuda\t[cuda]')
    parser.add_argument('--image_size', type=int, help='image size NxN \t[160]')
    parser.add_argument('--confidence', type=int, help='confidence thresholding, only higher conf predictions pass, if None, defaults to 0.25 (from YOLO docs) \t[0.25]')
    parser.add_argument('--batch_size', type=int, help='batch size for each YOLO inference step (speeds up processing significantly) \t[128]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    ### Assign defaults
    IN_DIR = args.data_dir or "data/stacked_segmentation"
    YOLO_DIR = args.yolo_dir or "yolo_checkpoint/weights/best.pt"
    OUT_DIR = args.out_dir or "heatmaps"
    DEVICE = args.device or "cuda"
    IMAGE_SIZE = args.image_size or 160
    CONFIDENCE = args.confidence or 0.25
    BATCH_SIZE = args.batch_size or 128
    WORKERS = args.workers or 10

    generate_heatmaps()
