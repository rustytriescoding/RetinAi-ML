import pandas as pd

from sklearn.model_selection import train_test_split

csv_path='../data/ocular-disease-recognition-odir5k/full_df.csv'
df = pd.read_csv(csv_path)

# Remove patient data if age < 5
df = df[df['Patient Age'] >= 5]

# Split the dataframes into 70% training, 15% validation, and 15% testing sets
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42) 
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)  

# Add code to check for duplicate file names in any of the dfs

# Save dataframes to CSV files
train_df.to_csv('./csvs/train.csv', index=False)
val_df.to_csv('./csvs/val.csv', index=False)
test_df.to_csv('./csvs/test.csv', index=False)
