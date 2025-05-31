# src/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
data_file = '../data/DataCoSupplyChainDataset.csv'
data = pd.read_csv(data_file, encoding='ISO-8859-1')

# Handle missing values (already done in the previous step)
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# For categorical columns, fill missing values with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Remove irrelevant columns
data = data.drop(columns=['Customer Password', 'Product Description'])

# Apply Label Encoding for columns with fewer categories
label_encoder = LabelEncoder()

# List of columns with fewer categories that we will encode using Label Encoding
label_encode_columns = ['Type', 'Delivery Status', 'Customer Segment', 'Market', 'Order Status', 'Shipping Mode']

for col in label_encode_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Apply One-Hot Encoding selectively (only for columns with fewer unique values)
# For example, 'Category Name' and 'Order Region' might have a smaller set of categories
one_hot_encode_columns = ['Category Name', 'Order Region', 'Order City']

data_encoded = pd.get_dummies(data, columns=one_hot_encode_columns, drop_first=True)

# Verify columns after encoding
print("\nColumns after encoding:")
print(data_encoded.columns)

# Scale the numeric columns using StandardScaler (already done in the previous step)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns
data_encoded[scaled_columns] = scaler.fit_transform(data_encoded[scaled_columns])

# Verify final preprocessed data
print("\nData after scaling:")
print(data_encoded.head())
