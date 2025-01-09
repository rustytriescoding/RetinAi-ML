import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import shutil

class DiscDatasetPrep:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        
        print(f"Available columns in CSV: {self.df.columns.tolist()}")
        
    def _create_label(self, seg_path):
        """Create YOLO format label from segmentation mask"""
        # Read the segmentation mask
        mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            return None
            
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Convert to YOLO format
        img_h, img_w = mask.shape
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        width = w / img_w
        height = h / img_h
        
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def _fix_path(self, path_str):
        """Fix path by removing duplicate 'optic-disc' folder reference"""
        parts = Path(path_str).parts
        # Find all occurrences of 'optic-disc'
        indices = [i for i, part in enumerate(parts) if part == 'optic-disc']
        
        if len(indices) > 1:
            # Keep only the first occurrence
            parts = parts[:indices[0]] + parts[indices[1]:]
        
        return Path(*parts)

    def prepare_dataset(self, output_dir, fundus_dir, seg_dir):
        """
        Prepare YOLO dataset - copy images and create labels
        
        Args:
            output_dir (str): Directory to save prepared dataset
            fundus_dir (str): Directory containing fundus images
            seg_dir (str): Directory containing segmentation masks
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        fundus_dir = Path(fundus_dir)
        seg_dir = Path(seg_dir)
        
        # Create directories
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize counters
        processed = 0
        skipped = 0
        
        # Process each image
        for idx, row in self.df.iterrows():
            try:
                # Skip if segmentation path is NaN
                if pd.isna(row['fundus_od_seg']):
                    print(f"Skipping row {idx}: Missing segmentation path")
                    skipped += 1
                    continue
                
                # Skip if fundus path is NaN
                if pd.isna(row['fundus']):
                    print(f"Skipping row {idx}: Missing fundus path")
                    skipped += 1
                    continue
                
                # Get segmentation mask path and fix it
                seg_rel_path = str(row['fundus_od_seg']).lstrip('/')
                seg_rel_path = self._fix_path(seg_rel_path)
                seg_path = seg_dir / seg_rel_path.name  # Only use the filename
                
                if not seg_path.exists():
                    print(f"Skipping row {idx}: Segmentation file not found at {seg_path}")
                    skipped += 1
                    continue
                
                # Create label from segmentation mask
                label = self._create_label(seg_path)
                if label is None:
                    print(f"Skipping row {idx}: Could not create label from {seg_path}")
                    skipped += 1
                    continue
                
                # Get corresponding fundus image path
                fundus_rel_path = str(row['fundus']).lstrip('/')
                fundus_path = fundus_dir / Path(fundus_rel_path).name  # Only use the filename
                
                if not fundus_path.exists():
                    print(f"Skipping row {idx}: Fundus image not found at {fundus_path}")
                    skipped += 1
                    continue
                
                # Save label
                label_path = labels_dir / f"{Path(fundus_rel_path).stem}.txt"
                with open(label_path, 'w') as f:
                    f.write(label)
                    
                # Copy fundus image
                shutil.copy2(str(fundus_path), str(images_dir / fundus_path.name))
                processed += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped += 1
                continue
        
        print(f"\nDataset preparation complete:")
        print(f"Processed: {processed} images")
        print(f"Skipped: {skipped} images")
        
        if processed == 0:
            raise Exception("No images were processed successfully. Please check your CSV file and paths.")

    def create_yaml(self, yaml_path, train_dir, val_dir):
        yaml_content = f"""
path: .  # Base path is current directory
train: {train_dir}  # Path to training images
val: {val_dir}  # Path to validation images

names:
  0: optic_disc
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
