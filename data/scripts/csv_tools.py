import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
import cv2
from sklearn.model_selection import train_test_split

class CsvTools():
    def __init__(self, dataset_csv, save_dir):
        self.dataset_csv = dataset_csv
        self.save_dir = save_dir

    def calculate_mean_std(self, image_path):
        mean = np.zeros(3)
        std = np.zeros(3)
        num_images = 0

        df = pd.read_csv(self.dataset_csv)

        for filename in tqdm(df['filename'], desc='Mean STD loop'):
            img_path = os.path.join(image_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            num_images += 1
            mean += img.mean(axis=(0, 1))
            std += img.std(axis=(0, 1))

        mean /= num_images
        std /= num_images

        mean /= 255.0
        std /= 255.0

        return mean.tolist(), std.tolist()

    def calculate_weights(self):
        df = pd.read_csv(self.dataset_csv)
        label_counts = df['labels'].value_counts()

        total = len(df)
        num_normal = label_counts.get("['N']", 0)
        num_glaucoma = label_counts.get("['G']", 0)

        # Inverse class frequency as weights
        weight_normal = total / (2 * num_normal)  # Weight for Normal (Class 0)
        weight_glaucoma = total / (2 * num_glaucoma)  # Weight for Glaucoma (Class 1)

        class_weights = torch.tensor([weight_normal, weight_glaucoma], dtype=torch.float32).to(device)
        return class_weights

    def clean_csv(self):
        df = pd.read_csv(self.dataset_csv)

        print('Original length of dataframe:', len(df))

        # Remove patient data if age < 5
        df = df[df['Patient Age'] >= 5]

        # Remove blurry images
        df = df[~(
            ((df['filename'].str.contains('left')) & (df['Left-Diagnostic Keywords'].str.contains('low image quality', case=False))) |
            ((df['filename'].str.contains('right')) & (df['Right-Diagnostic Keywords'].str.contains('low image quality', case=False)))
        )]

        # Remove other diagnosis N,D,G,C,A,H,M,O
        df = df[df['labels'] != "['C']"]
        df = df[df['labels'] != "['D']"]
        df = df[df['labels'] != "['A']"]
        df = df[df['labels'] != "['H']"]
        df = df[df['labels'] != "['M']"]
        df = df[df['labels'] != "['O']"]

        # Remove poor images
        bad_images = [
            "0_left.jpg",
            "418_left.jpg",
            "418_right.jpg",
            ]

        for bad_image in bad_images:
            df = df[df['filename'] != bad_image]

        print('Cleansed length of dataframe:', len(df))

        # Save cleaned csv
        df.to_csv('dataset1_clean.csv', index=False)
        
    def split_csv(self, train_split=0.8, test_split=None, csv=None):
        if not csv: 
            csv = self.dataset_csv

        df = pd.read_csv(csv)

        train_df, temp_df = train_test_split(df, train_size=train_split, random_state=42)  # Set random_state for reproducibility

        if test_split:
            val_df, test_df = train_test_split(temp_df, test_size=test_split, random_state=42)
            test_df.to_csv(os.path.join(self.save_dir, 'test.csv'), index=False)
        else:
            val_df = temp_df

        val_df.to_csv(os.path.join(self.save_dir, 'val.csv'), index=False)
        
        train_df.to_csv(os.path.join(self.save_dir, 'train.csv'), index=False)
        print('Split and saved csvs')

    def split_csv_5050(self, csv=None):
        if csv:
            df = pd.read_csv(csv)

            count_g = df[df['labels'] == "['G']"].shape[0]
            filtered_n = df[df['labels'] == "['N']"].sample(n=count_g, random_state=1)
            filtered_g = df[df['labels'] == "['G']"]

            final_df = pd.concat([filtered_n, filtered_g])
            final_df = final_df.sample(frac=1, random_state=1).reset_index(drop=True)
            final_df.to_csv(os.path.join(self.save_dir, 'split_5050.csv'), index=False)

    def split100(self, label, csv=None):
        if csv:
            df = pd.read_csv(csv)
            df = df[df['labels'] == f"['{label}']"]
            df.to_csv(os.path.join(self.save_dir, f'split_100_{label}.csv'), index=False)

def main():
    dataset1Tools = CsvTools(dataset_csv='../dataset1/csvs/dataset1.csv', save_dir='../dataset1/csvs')
    dataset1Tools.split_csv(train_split=0.7, test_split=0.5, csv='../dataset1/csvs/dataset1_clean.csv')

    dataset2Tools = CsvTools(dataset_csv='../dataset2/csvs/dataset2.csv', save_dir='../dataset2/csvs')
    dataset2Tools.split_csv(train_split=0.8)

if __name__ == '__main__':
    main()
