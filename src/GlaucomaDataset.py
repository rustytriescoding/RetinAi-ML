import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file) 
        self.transform = transform 
        self.root_dir = root_dir
        
        # Define a mapping from string labels to integers
        self.label_mapping = {
            "N": 0,  # Normal
            "G": 1,  # Glaucoma
        }

    def __len__(self):
        return len(self.df)  

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir,
                                self.df.filename[index])
        image = Image.open(image_path)
        
        label = self.df.labels[index]
        label = label.strip("[]").strip("'\"")  # Remove brackets from cell value
        
        # Convert the label to int
        label = self.label_mapping.get(label, -1)  # Use -1 for any unknown labels
        
        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
