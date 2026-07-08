import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_excel("student_dataset_balanced_.xlsx")

np.random.seed(42)

score = (
    df["Study_Hours"] * 4 +
    df["Attendance"] * 0.30 +
    df["Previous_Marks"] * 0.50 +
    df["Assignments"] * 2 +
    (8 - abs(df["Sleep_Hours"] - 7)) * 2 +
    np.random.normal(0, 5, len(df))
)

df["Result"] = np.where(score >= 70, 1, 0)

df.to_excel("student_dataset.xlsx", index=False)

print(df["Result"].value_counts())
print("Target column added successfully!")