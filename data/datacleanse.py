import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def calculate_class_weights(csv_file):
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

csv_path='../data/full_df.csv'
df = pd.read_csv(csv_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(len(df))
# Remove patient data if age < 5
df = df[df['Patient Age'] >= 5]
print(len(df))

# Remove blurry images
df = df[~(
    ((df['filename'].str.contains('left')) & (df['Left-Diagnostic Keywords'].str.contains('low image quality', case=False))) |
    ((df['filename'].str.contains('right')) & (df['Right-Diagnostic Keywords'].str.contains('low image quality', case=False)))
)]
print(len(df))

# Remove cataracts diagnosis
df = df[df['labels'] != "['C']"]
print(len(df))

# Remove other diagnosis N,D,G,C,A,H,M,O
df = df[df['labels'] != "['D']"]
df = df[df['labels'] != "['A']"]
df = df[df['labels'] != "['H']"]
df = df[df['labels'] != "['M']"]
df = df[df['labels'] != "['O']"]
print(len(df))

# Split the dataframes into 80% training, 10% validation, and 10% testing sets
train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42) 
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)  

# Save dataframes to CSV files
train_df.to_csv('./csvs/train.csv', index=False)
val_df.to_csv('./csvs/val.csv', index=False)
test_df.to_csv('./csvs/test.csv', index=False)

train_csv='../data/csvs/train.csv'
class_weights = calculate_class_weights(train_csv)
print(class_weights)
