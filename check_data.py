import pandas as pd

# Read the diabetes dataset
data = pd.read_csv("diabetes_binary_health.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display total rows and columns
print("\nDataset Shape:")
print(data.shape)

# Display all column names
print("\nColumn Names:")
print(data.columns)