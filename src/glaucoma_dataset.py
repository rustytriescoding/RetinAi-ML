import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class GlaucomaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, segment_dir=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations
            root_dir (str): Directory with all the full fundus images
            transform (callable, optional): Optional transform to be applied on images
            segment_dir (str, optional): Directory with segmented disc images
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        self.segment_dir = segment_dir
        self.use_segmented = segment_dir is not None
        
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
        
        # Verify all files exist
        self.validate_files()

    def validate_files(self):
        """Verify that all image files exist in both directories."""
        missing_files = []
        for idx, row in self.df.iterrows():
            filename = row[self.filename_col]
            full_path = os.path.join(self.root_dir, filename)
            
            if not os.path.exists(full_path):
                missing_files.append(('full', filename))
            
            if self.use_segmented:
                segment_path = os.path.join(self.segment_dir, filename)
                if not os.path.exists(segment_path):
                    missing_files.append(('segment', filename))
        
        if missing_files:
            print("Warning: Missing files detected:")
            for file_type, filename in missing_files:
                print(f"- {file_type}: {filename}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Returns both the full image and the segmented disc image if available
        """
        filename = self.df[self.filename_col][index]
        
        # Load full fundus image
        full_path = os.path.join(self.root_dir, filename)
        full_image = Image.open(full_path).convert('RGB')
        
        # Process the full image
        if self.transform:
            full_image = self.transform(full_image)
        
        # Load segmented disc image if available
        if self.use_segmented:
            segment_path = os.path.join(self.segment_dir, filename)
            segment_image = Image.open(segment_path).convert('RGB')
            if self.transform:
                segment_image = self.transform(segment_image)
        else:
            # If no segment directory provided, return the full image twice
            segment_image = full_image
        
        # Get label
        label = self.df['glaucoma'][index]
        label = torch.tensor(label, dtype=torch.long)
        
        return full_image, segment_image, label
