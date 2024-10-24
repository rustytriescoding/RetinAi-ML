import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

class FormatImage:
    def __init__(self, image_dir, save_dir):
        self.image_dir = image_dir
        self.save_dir = save_dir

    def adjust_gamma(self, image, gamma=0.4):
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def crop_and_save_images(self, gamma=0.4):
        os.makedirs(self.save_dir, exist_ok=True)

        for filename in tqdm(os.listdir(self.image_dir), desc='Image Edit Loop'):
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

            save_path = os.path.join(self.save_dir, filename)
            circular_image.save(save_path)

def main():
    dataset2Format = FormatImage(image_dir='../dataset2/raw', save_dir='../dataset2/processed')
    dataset2Format.crop_and_save_images()

if __name__ == '__main__':
    main()
