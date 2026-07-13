import pandas as pd
import pickle
import os

metadata_path = os.path.join("data", "books_metadata.pkl")
with open(metadata_path, "rb") as f:
    df_metadata = pickle.load(f)

raw_data_path = os.path.join("data", "books.csv") 
if not os.path.exists(raw_data_path):
    raw_data_path = os.path.join("data","raw","all_books_combined.csv")

if not os.path.exists(raw_data_path):
    print("Error: Original raw CSV file not found in 'data/' or 'data/raw/'.")
    exit()

df_raw = pd.read_csv(raw_data_path, usecols=["Name", "ISBN"])
df_raw = df_raw.drop_duplicates(subset=["Name"])

if "ISBN" in df_metadata.columns:
    df_metadata = df_metadata.drop(columns=["ISBN"])
    
df_updated = pd.merge(df_metadata, df_raw, on="Name", how="left")

with open(metadata_path, "wb") as f:
    pickle.dump(df_updated, f)
    
print("Success: ISBN injection completed successfully.")