"""
Pertemuan 3: Supervised Learning - Regression Models
Topic: Multiple Linear Regression, Evaluation Metrics (R2 & MSE)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("=== PERTEMUAN 3: LINEAR REGRESSION ===")

# Generate data sintesis
np.random.seed(42)
X = np.random.rand(100, 2) * 10
y = 2.5 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(100)

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi Performa
y_pred = model.predict(X_test)
print(f"Koefisien Model: {model.coef_}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R-Squared (R2) Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")
