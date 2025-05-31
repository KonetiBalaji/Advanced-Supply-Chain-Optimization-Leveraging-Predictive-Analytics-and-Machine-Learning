# src/data_exploration.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data_file = '../data/DataCoSupplyChainDataset.csv'
description_file = '../data/DescriptionDataCoSupplyChain.csv'
logs_file = '../data/tokenized_access_logs.csv'


# Load the datasets
data = pd.read_csv(data_file, encoding='ISO-8859-1')
description = pd.read_csv(description_file, encoding='ISO-8859-1')
logs = pd.read_csv(logs_file, encoding='ISO-8859-1')

# Show basic information about each dataset
print("Data Overview:")
print(data.info())
print("\nFirst 5 rows of the main data:")
print(data.head())  

print("\nDescription Data Overview:")
print(description.info())
print("\nFirst 5 rows of the description data:")
print(description.head())

print("\nLogs Data Overview:")
print(logs.info())
print("\nFirst 5 rows of the logs data:")
print(logs.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Check for duplicates
print("\nDuplicate rows in the dataset:", data.duplicated().sum())

# Basic statistical summary
print("\nStatistical Summary of the Data:")
print(data.describe())

print("\nColumns in the dataset:")
print(data.columns)


# Plot distribution of a key feature, e.g., demand
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], kde=True)
plt.title('Distribution of Sales (Demand)')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Filter only numeric columns for correlation
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
plt.show()
