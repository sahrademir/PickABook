import os
import re
import pandas as pd

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", "processed", "books_clean.csv")

    try:
        df= pd.read_csv(file_path)

        df["Name"] = df["Name"].fillna("")
        df["Description"] = df["Description"].fillna("")
        df["Authors"] = df["Authors"].fillna("")
         
        if len(df) > 50000:
            df=df.sample(n=50000, random_state=42).reset_index(drop=True)
        df= create_features(df)
        return df
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found! Please check the data cleaning step.")
        return None
    
def clean_text(text):
   text=text.lower()
   text= re.sub(r"[^a-zA-Z\s]", " ", text)
   text= re.sub(r"\s+", " ", text)

   return text.strip() 

def create_features(df):
    df["Features"]=(
        df["Name"]+ " "+
        df["Authors"]+ " "+
        df["Description"]
    )

    df["Features"] = df["Features"].apply(clean_text)
    return df