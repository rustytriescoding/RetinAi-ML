import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from pathlib import Path
from PIL import Image

class GlaucomaDataset3(Dataset):
    def __init__(self, csv_path, base_dir, transform=None):
        self.base_dir = base_dir
        self.csv_path = csv_path
        
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform
        
        # Convert -1 (suspicious) to 1 (glaucoma)
        self.df['types'] = self.df['types'].apply(lambda x: 1 if x == -1 else x)
        
        # Clean the fundus paths to only keep filename
        self.df['fundus'] = self.df['fundus'].apply(lambda x: Path(x).name if isinstance(x, str) else x)
        
        # Drop NaN entries
        self.df = self.df.dropna(subset=['fundus'])
        
        # Filter out missing images
        valid_images = []
        missing_count = 0
        
        for idx, row in self.df.iterrows():
            full_path = os.path.join(self.base_dir, row['fundus'])
            if os.path.exists(full_path):
                try:
                    # Try opening the image to verify it's valid
                    with Image.open(full_path) as img:
                        pass
                    valid_images.append(idx)
                except Exception as e:
                    missing_count += 1
                    print(f"Warning: Could not open image {full_path}: {e}")
            else:
                missing_count += 1
        
        # Keep only rows with valid images
        self.df = self.df.iloc[valid_images].reset_index(drop=True)
        
        if missing_count > 0:
            print(f"Warning: {missing_count} images were missing or invalid")
            print(f"Proceeding with {len(self.df)} valid images")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        fundus_path = self.df['fundus'].iloc[index]
        label = self.df['types'].iloc[index]
        
        full_path = os.path.join(self.base_dir, fundus_path)
        image = Image.open(full_path).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.float)
        
        return image, label

    def get_labels(self):
        return self.df['types'].values

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
        
        return full_image, label
