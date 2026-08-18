import pandas as pd
import os

CSV_PATH = r"C:\recipe_project\strong_dishes_recipe_dataset.csv"

print("=" * 60)
print("SMART RECIPE FINDER - DATASET TEST")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    print("ERROR: CSV file not found!")
    exit()

df = pd.read_csv(CSV_PATH)

print("\nCSV loaded successfully!")
print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset test completed successfully!")