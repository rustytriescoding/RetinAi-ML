import pandas as pd

def split5050(csv):
  df = pd.read_csv(f'csvs/{csv}.csv')

  count_g = df[df['labels'] == "['G']"].shape[0]

  filtered_n = df[df['labels'] == "['N']"].sample(n=count_g, random_state=1)

  filtered_g = df[df['labels'] == "['G']"]

  final_df = pd.concat([filtered_n, filtered_g])

  # Optional: Shuffle the final DataFrame
  final_df = final_df.sample(frac=1, random_state=1).reset_index(drop=True)

  # Step 5: Save the adjusted DataFrame back to CSV
  final_df.to_csv(f'/csvs/{csv}5050.csv', index=False)

def split100(label):
   df = pd.read_csv('csvs/test.csv')
   df = df[df['labels'] == f"['{label}']"]
   df.to_csv(f'csvs/testonly-{label}.csv', index=False)

def main():
  # split5050()
  split100('N')
  split100('G')






if __name__ == '__main__':
    main()
