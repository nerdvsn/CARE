import pandas as pd
import os

# Define the path to the data directory
data_dir = '/Users/gracialukelo/Desktop/Ba Koko/Kedorion/projects/Phoenix/Research/data'

# List all files in the data directory
files = os.listdir(data_dir)

# Read each file and print the first 3 rows
df = pd.read_parquet(os.path.join(data_dir, files[0]))
print(df.head(3).to_dict())
