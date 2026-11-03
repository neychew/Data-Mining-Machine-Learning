"""
Pertemuan 1: Introduction to Data Mining & Machine Learning Life Cycle
Topic: CRISP-DM Methodology & Data Loading from IBM Storage
"""

import pandas as pd
import numpy as np

# Load Sample Dataset dari IBM Storage
url = '[https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Datafiles/auto.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Datafiles/auto.csv)'
df = pd.read_csv(url, header=None)

# Menampilkan statistik dasar dan struktur data
print("Data Shape (Baris, Kolom):", df.shape)
print("\n5 Baris Pertama Data:")
print(df.head())

# Menampilkan informasi dasar dataset
print(f"Bentuk Dataset (Baris, Kolom): {df.shape}")
print("\n5 Baris Pertama Data:")
print(df.head())
