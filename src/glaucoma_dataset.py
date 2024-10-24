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
        
        # Determine CSV format and standardize it
        if 'labels' in self.df.columns:
            # Original format with N/G labels
            self.df['glaucoma'] = self.df['labels'].apply(lambda x: 1 if x.strip("[]'\"") == "G" else 0)
        elif 'Glaucoma' in self.df.columns:
            # New format with 0/1 labels
            self.df['glaucoma'] = self.df['Glaucoma']
        
        # Determine filename column
        if 'filename' in self.df.columns:
            self.filename_col = 'filename'
        elif 'Filename' in self.df.columns:
            self.filename_col = 'Filename'
        else:
            raise ValueError("CSV must contain either 'filename' or 'Filename' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.df[self.filename_col][index])
        image = Image.open(image_path)
        
        label = self.df['glaucoma'][index]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        
        return image, label
