"""
Pertemuan 6: Model Evaluation, Cross-Validation, & Hyperparameter Tuning
Topic: Grid Search CV dengan Support Vector Machine (SVM)
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

print("=== PERTEMUAN 6: GRID SEARCH CV & SVM ===")

# Load Data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Menentukan Parameter Grid untuk Tuning SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Inisialisasi GridSearch dengan 5-Fold Cross Validation
grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Hyperparameter Terbaik: {grid_search.best_params_}")
print(f"Akurasi Terbaik pada Cross-Validation: {grid_search.best_score_:.4f}")

# Evaluasi pada Data Testing
test_acc = grid_search.score(X_test, y_test)
print(f"Akurasi pada Hold-Out Test Set: {test_acc:.4f}")
