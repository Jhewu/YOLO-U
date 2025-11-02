from custom_yolo_predictor.custom_detseg_predictor import CustomDetectionPredictor

import torch
from torchvision.utils import save_image
from torchvision import transforms

from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import argparse
import os

def generate_objectmaps(): 
    """
    TODO: REFRASE
    
    Generate heatmaps from YOLO Prediction Result object and saves
    it in the destination directory. Refer to the if __name__ == "__main__"
    for information about the global function parameters
    """

    """------START OF HELPER FUNCTIONS------"""
    def create_dir(path:str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        
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
    objectmap_dest_dir    =  os.path.join(OUT_DIR, "objectmap")

    # Declare args for custom YOLO and extract model itself
    args = dict(save=False, verbose=False, device=DEVICE, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, conf=CONFIDENCE) # MIGHT REMOVE SOME UNUSED ARGUMENTS
    predictor = CustomDetectionPredictor(overrides=args)
    predictor.setup_model(YOLO_DIR)

    model = predictor.model.model ; model.to(DEVICE)

    # Create transform
    transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()])

    for split in ["test", "train", "val"]: 
        ## Directories are structured as 'dataset/images/split/image1, image2,..., imagen'
        image_split         = os.path.join(image_dir, split)    
        objectmap_split       = os.path.join(objectmap_dest_dir, split)
        create_dir(objectmap_split)

        ## Create full paths of images. The line below does the following: 
        # (1) Gets the list of images
        # (2) Sort the images (ensure images order)
        # (3) Create the full paths of each image
        image_full_paths = [os.path.join(image_split, image) for image in sorted( os.listdir(image_split) )]

        ### ----------------------------------------
        # Single Batch | Inference Per Images
        for image_path in image_full_paths[:]: 
            image_tensor = transform(Image.open(image_path).convert("RGBA")) ; image_tensor = image_tensor.to(DEVICE)
            result = model(image_tensor.unsqueeze(0))

            # Unpack and obtain the CLS object maps
            detect_branch, cls_branch = result
            twenty, ten, five = cls_branch # 20x20, 10x10, 5x5

            # Apply Sigmoid
            # twenty, ten = torch.sigmoid(twenty[:, -1:]), torch.sigmoid(ten[:, -1:]) # <- Obtain the last channel
            twenty, ten = twenty[:, -1:], ten[:, -1:] # <- Obtain the last channel

            # Create dest filename
            basename = os.path.basename(image_path)
            filename = basename.split(".")[0] # <- Get filename without png extension
            
            twenty_dest_dir = os.path.join(objectmap_split, f"{filename}_20.png")
            ten_dest_dir = os.path.join(objectmap_split, f"{filename}_10.png")

            print(f"Saving... {twenty_dest_dir}")
            print(f"Saving... {ten_dest_dir}")
            
            save_image(twenty, twenty_dest_dir) ; save_image(ten, ten_dest_dir)

        ### ----------------------------------------
        # Multi Batch | Multiple Inference Per Batch (Multi-Images)
        ### TODO: WORK ON MULTI-BATCH LATER
#         batches = [image_full_paths[i:i + BATCH_SIZE] for i in range(0, len(image_full_paths), BATCH_SIZE)]
# 
#         with ThreadPoolExecutor(max_workers=WORKERS) as executor: 
#             futures = []
#             for batch in batches: 
#                 batch_results = predictor(batch)
#                 for result in batch_results: 
#                     # Submit tasks and store the future
#                     future = executor.submit(generate_heatmaps_from_bbox, [result], heatmap_split)
#                     futures.append(future)
#             # Wait for all futures to complete before exiting the with block
#             for future in as_completed(futures): 
#                 try: 
#                     future.result() # This will raise any exceptions that occurred in the thread
#                 except Exception as e: 
#                     print(f"Error processing heatmap: {e}")
        ### ----------------------------------------

if __name__ == "__main__": 
    # ---------------------------------------------------
    des="""
    Using a pre-trained YOLO Ultralytics model (YOLOv12n), 
    this script generates "object maps" from YOLOv12n CLS head, 
    where pixel intensity represent the confidence of an object 
    in the location. Object maps will will be saved as a separate 
    directory (to not modified original preprocessed images). 
    """
    # ---------------------------------------------------

    ### Parse arguments
    parser = argparse.ArgumentParser(description=des.lstrip(" "), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data_dir", type=str,help='directory (root) of BraTS dataset\t[stacked_segmentation]')
    parser.add_argument("--yolo_dir", type=str,help='directory of YOLO Ultralytics model weights\t[yolo_checkpoint/weights/best.pt]')
    parser.add_argument('--out_dir',type=str,help='output directory of the generated object maps\t[object_map]')
    parser.add_argument("--device", type=str,help='cpu or cuda\t[cuda]')
    parser.add_argument('--image_size', type=int, help='image size NxN \t[160]')
    parser.add_argument('--confidence', type=int, help='confidence thresholding, only higher conf predictions pass, if None, defaults to 0.25 (from YOLO docs) \t[0.25]')
    parser.add_argument('--batch_size', type=int, help='batch size for each YOLO inference step (speeds up processing significantly) \t[128]')
    parser.add_argument('--workers', type=int, help='number of threads/workers to use\t[10]')
    args = parser.parse_args()

    ### Assign defaults
    IN_DIR = args.data_dir or "data/stacked_segmentation"
    YOLO_DIR = args.yolo_dir or "yolo_checkpoint/weights/best.pt"
    OUT_DIR = args.out_dir or "data/stacked_segmentation"
    DEVICE = args.device or "cuda"
    IMAGE_SIZE = args.image_size or 160
    CONFIDENCE = args.confidence or 0.25
    BATCH_SIZE = args.batch_size or 128
    WORKERS = args.workers or 10

    generate_objectmaps()

    """
    TODO: 
    (1) Remove uncessessary parameters
    (2) Implement batch processing with a custom dataset
    (3) Redo documentation

    """
