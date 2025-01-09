import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class DatasetStatistics:
    def __init__(self, image_dir, image_size=224, channel_mode='rgb'):
        """
        Initialize the dataset statistics calculator.
        
        Args:
            image_dir (str): Directory containing the images
            image_size (int): Size to resize images to
            channel_mode (str): One of 'rgb', 'green', or 'grayscale'
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.channel_mode = channel_mode
        
        # Base transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        
        self.transform = transforms.Compose(transform_list)
        
    def _process_channels(self, tensor):
        """Process tensor according to channel mode"""
        if self.channel_mode == 'green':
            # Extract green channel and replicate it 3 times for compatibility
            return tensor[1].unsqueeze(0).repeat(3, 1, 1)
        elif self.channel_mode == 'grayscale':
            return tensor.mean(dim=0).unsqueeze(0).repeat(3, 1, 1)
        return tensor  # RGB mode

    def visualize_channels(self, image_path):
        """
        Visualize RGB channels and green channel of a sample image.
        """
        # Read image
        image = Image.open(image_path)
        tensor = self.transform(image)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Channel Comparison', fontsize=16)
        
        # Original RGB
        axes[0, 0].imshow(tensor.permute(1, 2, 0))
        axes[0, 0].set_title('Original RGB')
        
        # Red channel
        axes[0, 1].imshow(tensor[0], cmap='gray')
        axes[0, 1].set_title('Red Channel')
        
        # Green channel
        axes[1, 0].imshow(tensor[1], cmap='gray')
        axes[1, 0].set_title('Green Channel')
        
        # Blue channel
        axes[1, 1].imshow(tensor[2], cmap='gray')
        axes[1, 1].set_title('Blue Channel')
        
        plt.tight_layout()
        plt.show()

    def calculate_stats(self):
        """
        Calculate mean and standard deviation of the dataset.
        """
        image_files = [f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            raise Exception(f"No images found in {self.image_dir}")
        
        pixel_sum = 0
        pixel_sum_squared = 0
        num_pixels = 0
        
        print(f"Calculating dataset statistics ({self.channel_mode} mode)...")
        for filename in tqdm(image_files):
            image_path = os.path.join(self.image_dir, filename)
            try:
                image = Image.open(image_path)
                tensor = self.transform(image)
                
                # Process channels according to mode
                tensor = self._process_channels(tensor)
                
                # Update sums
                pixel_sum += torch.sum(tensor, dim=[1, 2])
                pixel_sum_squared += torch.sum(tensor ** 2, dim=[1, 2])
                num_pixels += tensor.shape[1] * tensor.shape[2]
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        mean = (pixel_sum / num_pixels).numpy()
        std = (torch.sqrt(pixel_sum_squared / num_pixels - (pixel_sum / num_pixels) ** 2)).numpy()
        
        return mean, std
    
    def print_stats(self):
        """
        Calculate and print the dataset statistics in a formatted way.
        """
        mean, std = self.calculate_stats()
        
        print(f"\nDataset Statistics ({self.channel_mode.upper()} mode):")
        print("-" * 50)
        print("Mean:")
        for i, m in enumerate(mean):
            print(f"Channel {i}: {m:.3f}")
        print("\nStandard Deviation:")
        for i, s in enumerate(std):
            print(f"Channel {i}: {s:.3f}")
        print("-" * 50)
        
        print("\nFor use in transforms.Normalize():")
        mean_str = "[" + ", ".join([f"{m:.3f}" for m in mean]) + "]"
        std_str = "[" + ", ".join([f"{s:.3f}" for s in std]) + "]"
        print(f"transforms.Normalize(mean={mean_str}, std={std_str})")

def main():
    # Initialize statistics calculator
    stats_calculator = DatasetStatistics(
        image_dir='../dataset1/disc-crop',
        image_size=224,
        channel_mode='green'  # Options: 'rgb', 'green', 'grayscale'
    )
    
    # Visualize channels of first image (optional)
    first_image = '../dataset1/disc-crop/7_right.jpg'
    stats_calculator.visualize_channels(first_image)
    
    # Calculate and print statistics
    # stats_calculator.print_stats()

if __name__ == "__main__":
    main()
