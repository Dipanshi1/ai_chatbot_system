import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ml.config import RAW_DATA_PATH

df = pd.read_excel(RAW_DATA_PATH)

print(df.head())
print(df.shape)
print(df.columns)
df.info()

print(df.isnull().sum())
print(df["Intent"].value_counts())
print(df["Intent"].nunique())
print(df["Topic"].value_counts())
print(df["User Message"].head())
print(df["User Message"].str.len().describe())
print(df["User Message"].duplicated().sum())
print("\n--- Sentiment Distribution ---")
print(df["Sentiment Label"].value_counts())

print("\n--- Topic Count ---")
print(df["Topic"].nunique())

print("\n--- Intent Proportions ---")
print(df["Intent"].value_counts(normalize=True).head())

print("\n--- User Message Length ---")
print(df["User Message"].str.len().describe())

print("\n--- Duplicate Messages ---")
print(df["User Message"].duplicated().sum())