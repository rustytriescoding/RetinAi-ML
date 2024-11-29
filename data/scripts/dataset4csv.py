import os
import pandas as pd

def create_glaucoma_dataset_csv(glaucoma_dir, non_glaucoma_dir, output_csv='dataset4.csv'):
    # Initialize lists to store data
    image_files = []
    labels = []
    
    # Process glaucoma images (label 1)
    for img_file in os.listdir(glaucoma_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(img_file)
            labels.append(1)
    
    # Process non-glaucoma images (label 0)
    for img_file in os.listdir(non_glaucoma_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(img_file)
            labels.append(0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'fundus': image_files,
        'types': labels
    })
    
    # Save CSV file
    df.to_csv('../dataset4/csvs/dataset4.csv', index=False)
    print(f"Created CSV file with {len(df)} images")
    print(f"Number of glaucoma images: {len(df[df['types'] == 1])}")
    print(f"Number of non-glaucoma images: {len(df[df['types'] == 0])}")

# Example usage
if __name__ == "__main__":
    glaucoma_dir = "../../tempdataset/LAG_database_part_1/suspicious_glaucoma/image"
    non_glaucoma_dir = "../../tempdataset/LAG_database_part_1/non_glaucoma/image"
    
    create_glaucoma_dataset_csv(glaucoma_dir, non_glaucoma_dir)
