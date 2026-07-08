import pandas as pd

# Load dataset
df = pd.read_excel("student_dataset_balanced_.xlsx")

print("Missing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Check data types
print("\nData Types:")
print(df.dtypes)

# Save cleaned dataset
df.to_excel("student_dataset.xlsx", index=False)

print("\nData cleaning completed successfully!")