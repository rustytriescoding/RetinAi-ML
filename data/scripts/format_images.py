import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

class FormatImage:
    def __init__(self, image_dir, processed_dir, segment_dir):
        self.image_dir = image_dir
        self.processed_dir = processed_dir
        self.segment_dir = segment_dir

    def adjust_gamma(self, image, gamma=0.3):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def crop_disc(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(l_channel)
        
        # Create masks at different brightness levels
        masks = []
        for percentile in [98, 95, 90]:
            thresh = np.percentile(enhanced, percentile)
            _, mask = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)
            masks.append(mask.astype(np.uint8))
        
        # Try each mask until we find a good disc candidate
        height, width = image.shape[:2]
        image_center = np.array([width/2, height/2])
        min_disc_size = min(width, height) * 0.05  # Minimum disc size (5% of image)
        max_disc_size = min(width, height) * 0.2   # Maximum disc size (20% of image)
        
        best_contour = None
        best_score = -float('inf')
        
        for mask in masks:
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate basic metrics
                area = cv2.contourArea(contour)
                
                # Skip if too small or too large
                if area < np.pi * (min_disc_size/2)**2 or area > np.pi * (max_disc_size/2)**2:
                    continue
                    
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4*np.pi*area/(perimeter*perimeter) if perimeter > 0 else 0
                
                # Calculate center and distance from image center
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center_dist = np.linalg.norm(np.array([cx, cy]) - image_center)
                
                # Calculate average brightness in the region
                mask = np.zeros_like(l_channel)
                cv2.drawContours(mask, [contour], -1, (255), -1)
                mean_brightness = cv2.mean(l_channel, mask=mask)[0]
                
                # Normalize and combine scores
                dist_score = 1 - (center_dist / (width/2))  # Higher score closer to center
                brightness_score = mean_brightness / 255
                
                # Combined score with weights
                score = (0.3 * circularity + 
                        0.3 * dist_score + 
                        0.4 * brightness_score)  # Increased weight for brightness
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
            
            # If we found a good candidate, stop trying more masks
            if best_score > 0.6:
                break
        
        if best_contour is None:
            # Fallback: if no disc detected, crop center of image
            size = min(width, height) // 3
            x1 = (width - size) // 2
            y1 = (height - size) // 2
            cropped = image[y1:y1+size, x1:x1+size]
            return cropped
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Calculate square crop size (max of width and height plus padding)
        crop_size = int(max(w, h) * 1.5)  # 50% padding
        
        # Ensure minimum size
        min_size = min(width, height) // 4
        crop_size = max(crop_size, min_size)
        
        # Calculate crop center
        center_x = x + w//2
        center_y = y + h//2
        
        # Calculate crop boundaries
        x1 = center_x - crop_size//2
        y1 = center_y - crop_size//2
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # Adjust if out of bounds
        if x1 < 0:
            x1 = 0
            x2 = crop_size
        if y1 < 0:
            y1 = 0
            y2 = crop_size
        if x2 > width:
            x2 = width
            x1 = width - crop_size
        if y2 > height:
            y2 = height
            y1 = height - crop_size
        
        # Ensure we stay within image bounds
        x1 = max(0, min(x1, width - crop_size))
        y1 = max(0, min(y1, height - crop_size))
        x2 = min(width, x1 + crop_size)
        y2 = min(height, y1 + crop_size)
        
        # Crop image
        cropped = image[y1:y2, x1:x2]
        
        return cropped

    def crop_disc_and_save_images(self):
        os.makedirs(self.segment_dir, exist_ok=True)
        
        failed_detections = []
        
        for filename in tqdm(os.listdir(self.image_dir), desc='Image Segment Loop'):
            image_path = os.path.join(self.image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {filename}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            try:
                cropped_image = self.crop_disc(image)
                
                if cropped_image is not None:
                    save_path = os.path.join(self.segment_dir, filename)
                    cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, cropped_image_bgr)
                else:
                    failed_detections.append(filename)
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                failed_detections.append(filename)
        
        if failed_detections:
            print(f"\nFailed to detect disc in {len(failed_detections)} images:")
            for fname in failed_detections:
                print(f"- {fname}")


    def crop_and_save_images(self, gamma=0.3):
        os.makedirs(self.processed_dir, exist_ok=True)

        for filename in tqdm(os.listdir(self.image_dir), desc='Image Crop Loop'):
            image_path = os.path.join(self.image_dir, filename)
            image = Image.open(image_path).convert("RGB")  
            image_np = np.array(image)
            
            # Apply gamma correction
            image_np = self.adjust_gamma(image_np, gamma)

            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Crop the image to the bounding rectangle
                circular_image = image_np[y:y+h, x:x+w]
            else:
                circular_image = image_np  

            circular_image = Image.fromarray(circular_image)

            save_path = os.path.join(self.processed_dir, filename)
            circular_image.save(save_path)

def main():
    dataset1Format = FormatImage(image_dir='../dataset1/processed', segment_dir='../dataset1/segment', processed_dir='../dataset1/processed')
    # dataset2Format.crop_and_save_images(gamma=0.3)
    dataset1Format.crop_disc_and_save_images()

if __name__ == '__main__':
    main()
