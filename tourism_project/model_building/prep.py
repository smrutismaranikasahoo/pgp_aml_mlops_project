import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from the repository data folder
df = pd.read_csv("tourism_project/data/tourism.csv")

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove unnecessary column
df = df.drop(columns=["CustomerID"], errors="ignore")

print("Cleaned dataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Training data shape:", Xtrain.shape)
print("Testing data shape:", Xtest.shape)
print("Train and test datasets saved successfully.")
