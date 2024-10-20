import pandas as pd

from sklearn.model_selection import train_test_split

csv_path='../data/full_df.csv'
df = pd.read_csv(csv_path)

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
