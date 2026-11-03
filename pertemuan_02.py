"""
Pertemuan 2: Data Preprocessing, Cleaning, & Exploratory Data Analysis (EDA)
Topic: Handling Missing Values, Feature Scaling, and Correlation Matrix
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

print("=== PERTEMUAN 2: PREPROCESSING & EDA ===")

# Buat dummy dataset untuk simulasi preprocessing
np.random.seed(42)
data = {
    'Age': [25, 30, np.nan, 35, 40, 22, 38],
    'Income': [50000, 60000, 55000, np.nan, 80000, 45000, 72000],
    'Score': [85, 90, 78, 88, 95, 70, 92]
}
df = pd.DataFrame(data)

# 1. Handling Missing Values (Imputasi Mean)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Income'].fillna(df['Income'].median(), inplace=True)

print("Data setelah imputasi missing values:")
print(df)

# 2. Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_features, columns=df.columns)

print("\nData setelah Standardization (Z-score):")
print(df_scaled.head())

# 3. Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrix Korelasi Fitur")
plt.show()
