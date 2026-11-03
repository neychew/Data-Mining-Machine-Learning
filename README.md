# 📊 Data Mining & Machine Learning - Academic Portfolio

Selamat datang di repositori dokumentasi pembelajaran dan praktikum mata kuliah **Data Mining & Machine Learning**. Repositori ini berisi rangkuman materi teori, eksplorasi kode Python, serta implementasi algoritma yang diadaptasi dari kurikulum **IBM Data Science Specialization (Coursera)** dan materi perkuliahan.

---

## 📑 Daftar Isi Pertemuan

1. [Pertemuan 1: Introduction to Data Mining & Machine Learning Life Cycle](#pertemuan-1-introduction-to-data-mining--machine-learning-life-cycle)
2. [Pertemuan 2: Data Preprocessing, Cleaning, & Exploratory Data Analysis (EDA)](#pertemuan-2-data-preprocessing-cleaning--exploratory-data-analysis-eda)
3. [Pertemuan 3: Supervised Learning - Regression Models](#pertemuan-3-supervised-learning---regression-models)
4. [Pertemuan 4: Supervised Learning - Classification Algorithms](#pertemuan-4-supervised-learning---classification-algorithms)
5. [Pertemuan 5: Unsupervised Learning - Clustering & Dimensionality Reduction](#pertemuan-5-unsupervised-learning---clustering--dimensionality-reduction)
6. [Pertemuan 6: Model Evaluation, Cross-Validation, & Hyperparameter Tuning](#pertemuan-6-model-evaluation-cross-validation--hyperparameter-tuning)

---

## 📌 Pertemuan 1: Introduction to Data Mining & Machine Learning Life Cycle

### 💡 Konsep & Teori
* **Metodologi CRISP-DM** (*Cross-Industry Standard Process for Data Mining*): Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment.
* **Perbedaan Supervised, Unsupervised, & Reinforcement Learning**.

### 💻 Implementasi Kode (Setting Up Environment & Data Overview)
```python
import pandas as pd
import numpy as np

# Load Sample Dataset (IBM Telecom Customer Churn Concept)
df = pd.read_csv('[https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Datafiles/auto.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Datafiles/auto.csv)', header=None)

# Inspection
print("Data Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
