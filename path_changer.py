import os
import pandas as pd

# Define old and new path prefixes
old_prefix = 'C:/Users/Administrator/Desktop/TCT_data/data/JPEGImages/'
new_prefix = 'C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/data/JPEGImages/'

# Root directory containing the csv files
root_dir = 'C:/Users/Administrator/Desktop/Cervical-Cancer-Detection/csvfiles'

# Walk through all folders and process csv files
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.csv'):
            csv_path = os.path.join(subdir, file)

            # Read CSV
            df = pd.read_csv(csv_path)

            # Replace the prefix in image_path column
            df['image_path'] = df['image_path'].str.replace(old_prefix, new_prefix, regex=False)

            # Overwrite the CSV with updated paths
            df.to_csv(csv_path, index=False)
            print(f'Updated: {csv_path}')