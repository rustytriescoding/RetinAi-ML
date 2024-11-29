from pathlib import Path
from sys import path
path.append('../../src')
from yolo_model import DiscDetectionModel
import cv2
from tqdm import tqdm
import os
import numpy as np

class DiscCropper:
    def __init__(self, weights_path='runs/detect/train/weights/best.pt', conf_threshold=0.5, padding_percent=0.2):
        """
        Initialize the DiscCropper with model weights and parameters.
        
        Args:
            weights_path (str): Path to the YOLO model weights
            conf_threshold (float): Confidence threshold for detections
            padding_percent (float): Percentage of padding to add around the detection
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.padding_percent = padding_percent
        self.model = DiscDetectionModel(model_name=str(self.weights_path))
    
    def _make_square_coords(self, x1, y1, x2, y2, img_shape):
        """
        Convert rectangular coordinates to square coordinates with padding.
        
        Args:
            x1, y1, x2, y2 (int): Original bounding box coordinates
            img_shape (tuple): Shape of the original image (height, width)
        
        Returns:
            tuple: New square coordinates (x1, y1, x2, y2)
        """
        # Calculate center and current size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        current_width = x2 - x1
        current_height = y2 - y1
        
        # Use the larger dimension for the square
        size = max(current_width, current_height)
        
        # Add padding
        padding = int(size * self.padding_percent)
        size_with_padding = size + (2 * padding)
        
        # Calculate new coordinates
        new_x1 = center_x - (size_with_padding // 2)
        new_y1 = center_y - (size_with_padding // 2)
        new_x2 = new_x1 + size_with_padding
        new_y2 = new_y1 + size_with_padding
        
        # Ensure coordinates are within image bounds
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_shape[1], new_x2)
        new_y2 = min(img_shape[0], new_y2)
        
        return new_x1, new_y1, new_x2, new_y2
    
    def crop_image(self, image_path, output_path):
        """
        Detect optic disc, crop it with padding, and save as a square image.
        
        Args:
            image_path (str or Path): Path to input image
            output_path (str or Path): Path to save cropped image
        
        Returns:
            bool: True if successful, False otherwise
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not read image at {image_path}")
            return False
            
        # Get model predictions
        results = self.model.predict(str(image_path), conf=self.conf_threshold)
        
        # Process highest confidence detection
        best_detection = None
        best_conf = -1
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0]
                    best_detection = (int(x1), int(y1), int(x2), int(y2))
        
        if best_detection is None:
            print(f"No optic disc detected in {image_path}")
            return False
        
        # Get square coordinates with padding
        x1, y1, x2, y2 = self._make_square_coords(*best_detection, img.shape)
        
        # Crop image
        cropped = img[y1:y2, x1:x2]
        
        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save cropped image
        cv2.imwrite(str(output_path), cropped)
        return True

def main():
    # Configuration
    # input_dir = Path('../dataset3/raw')
    # output_dir = Path('../dataset3/disc-crop')
    input_dir = Path('../dataset1/raw')
    output_dir = Path('../dataset1/disc-crop')
    weights_path = '../../models/yolo/retinai_yolo.pt'
    
    # Initialize cropper
    cropper = DiscCropper(
        weights_path=weights_path,
        conf_threshold=0.90, #higher to reduce poor images. 66 bad 0.5, 238 bad 0.85, 1357 bad 0.90
        padding_percent=0.5 # Increase from 0.2
    )
    
    processed = 0
    failed = 0
    
    for filename in os.listdir(input_dir):
        
        filepath = input_dir / filename
        output_path = output_dir / filename

        if cropper.crop_image(filepath, output_path):
            processed += 1
        else:
            failed += 1

    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Failed: {failed} images")

if __name__ == "__main__":
    main()
