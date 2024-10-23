import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def adjust_gamma(image, gamma=0.4):
  inv_gamma = 1.0 / gamma
  table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
  return cv2.LUT(image, table)

def crop_and_save_images(source_folder, target_folder, gamma=0.4):
  os.makedirs(target_folder, exist_ok=True)
    
  for filename in tqdm(os.listdir(source_folder), desc='Image Edit Loop'):
      if filename.endswith(('.png', '.jpg', '.jpeg')):  
        image_path = os.path.join(source_folder, filename)
        image = Image.open(image_path).convert("RGB")  
        image_np = np.array(image)
        
        # Apply gamma correction
        image_np = adjust_gamma(image_np, gamma)

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

        save_path = os.path.join(target_folder, filename)
        circular_image.save(save_path)

source_dir = 'ODIR-5K/Training Images'
target_dir = 'Formatted Images'   
crop_and_save_images(source_dir, target_dir)
