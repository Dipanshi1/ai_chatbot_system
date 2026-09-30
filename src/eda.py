import pandas as pd

df = pd.read_excel("data/raw/AI-Powered Chatbot.xlsx")

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