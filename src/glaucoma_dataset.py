import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, segment_dir=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        self.segment_dir = segment_dir
        self.use_segmented = segment_dir is not None
        
        if 'labels' in self.df.columns:
            self.df['glaucoma'] = self.df['labels'].apply(lambda x: 1 if x.strip("[]'\"") == "G" else 0)
        elif 'Glaucoma' in self.df.columns:
            self.df['glaucoma'] = self.df['Glaucoma']
        
        if 'filename' in self.df.columns:
            self.filename_col = 'filename'
        elif 'Filename' in self.df.columns:
            self.filename_col = 'Filename'
        else:
            raise ValueError("CSV must contain either 'filename' or 'Filename' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df[self.filename_col][index]
        
        full_path = os.path.join(self.root_dir, filename)
        full_image = Image.open(full_path).convert('RGB')
        
        if self.transform:
            full_image = self.transform(full_image)
        
        if self.use_segmented:
            segment_path = os.path.join(self.segment_dir, filename)
            segment_image = Image.open(segment_path).convert('RGB')
            if self.transform:
                segment_image = self.transform(segment_image)
        else:
            segment_image = full_image
        
        label = self.df['glaucoma'][index]
        label = torch.tensor(label, dtype=torch.long)
        
        return full_image, segment_image, label
