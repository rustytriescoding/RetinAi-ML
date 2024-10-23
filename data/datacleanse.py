import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
import cv2
from sklearn.model_selection import train_test_split

def calculateMeanStd(csv, image_path):
  mean = np.zeros(3)
  std = np.zeros(3)
  num_images = 0

  df = pd.read_csv(csv)

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

def calculateClassWeights(csv_file):
  df = pd.read_csv(csv_file)
  label_counts = df['labels'].value_counts()

  total = len(df)
  num_normal = label_counts.get("['N']", 0)
  num_glaucoma = label_counts.get("['G']", 0)

  # Inverse class frequency as weights
  weight_normal = total / (2 * num_normal)  # Weight for Normal (Class 0)
  weight_glaucoma = total / (2 * num_glaucoma)  # Weight for Glaucoma (Class 1)

  class_weights = torch.tensor([weight_normal, weight_glaucoma], dtype=torch.float32).to(device)
  return class_weights

def cleanCsv():
  csv_path='../data/full_df.csv'
  df = pd.read_csv(csv_path)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  print('Original length of dataframe:', len(df))
  # Remove patient data if age < 5
  df = df[df['Patient Age'] >= 5]

  # Remove blurry images
  df = df[~(
      ((df['filename'].str.contains('left')) & (df['Left-Diagnostic Keywords'].str.contains('low image quality', case=False))) |
      ((df['filename'].str.contains('right')) & (df['Right-Diagnostic Keywords'].str.contains('low image quality', case=False)))
  )]

  # Remove cataracts diagnosis
  df = df[df['labels'] != "['C']"]

  # Remove other diagnosis N,D,G,C,A,H,M,O
  df = df[df['labels'] != "['D']"]
  df = df[df['labels'] != "['A']"]
  df = df[df['labels'] != "['H']"]
  df = df[df['labels'] != "['M']"]
  df = df[df['labels'] != "['O']"]

  # Remove bad images
  bad_images = [
      "0_left.jpg",
      "418_left.jpg",
      "418_right.jpg",
    ]

  for bad_image in bad_images:
    df = df[df['filename'] != bad_image]

  print('Cleansed length of dataframe:', len(df))
  
  # Split the dataframes into 80% training, 10% validation, and 10% testing sets
  train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42) 
  val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)  

  # Save dataframes to CSV files
  df.to_csv('full_df_clean.csv', index=False)
  train_df.to_csv('./csvs/train.csv', index=False)
  val_df.to_csv('./csvs/val.csv', index=False)
  test_df.to_csv('./csvs/test.csv', index=False)
  print('Saved to csvs')

def calculateClassWeights2():
  num_normal = 386
  num_glaucoma = 134

  total = num_glaucoma + num_normal

  # Inverse class frequency as weights
  weight_normal = total / (2 * num_normal)  # Weight for Normal (Class 0)
  weight_glaucoma = total / (2 * num_glaucoma)  # Weight for Glaucoma (Class 1)

  class_weights = torch.tensor([weight_normal, weight_glaucoma], dtype=torch.float32)
  print(class_weights)

def main():
  # calculateClassWeights2()
  
  # Class weights
  # train_csv='../data/csvs/train.csv'
  # class_weights = calculate_class_weights(train_csv)
  # print(class_weights)

  cleanCsv()
  # Calculate Mean and Standard deviation of training images 
  train_csv='csvs/train.csv'
  image_path='Cropped Images'
  mean, std = calculateMeanStd(train_csv, image_path)
  print(f'Mean: {mean}, Std: {std}')

if __name__ == '__main__':
  main()
