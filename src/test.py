import pandas as pd

df = pd.read_csv(r"F:\DoAn\datasets\VSASV_20000\labels.csv")

print(len(df))
print(df["binary_label"].value_counts())
print(df["original_type"].value_counts())
print(df["split"].value_counts())