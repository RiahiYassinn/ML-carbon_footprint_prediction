# Regression Models with Hyperparameter Tuning

This repository contains implementations of three popular regression models using **scikit-learn**:  
- K-Nearest Neighbors Regressor (KNN)  
- Support Vector Regressor (SVR)  
- Decision Tree Regressor  

Each model is trained and tuned using **GridSearchCV** to find the best hyperparameters based on minimizing the mean squared error (MSE). The models are then evaluated on a test set with metrics such as MSE, RMSE, and R² score.

---

## Features

- Hyperparameter tuning with cross-validation for improved model performance
- Performance evaluation with standard regression metrics
- Modular code for training and evaluation of each model
- Supports parallel processing via `n_jobs=-1` for faster grid search

---

## Files and Functions

### K-Nearest Neighbors (KNN)
- **Function:** `train_knn_model(X_train, y_train, cv=5)`  
  Trains a KNN regressor using GridSearchCV over different numbers of neighbors.
- **Evaluation:** `evaluate_model(model, X_test, y_test)` prints and returns MSE, RMSE, and R².

### Support Vector Regressor (SVR)
- **Function:** `train_svr_model(X_train, y_train, cv=5)`  
  Trains an SVR model tuning hyperparameters like `C`, `gamma`, and `kernel`.
- **Evaluation:** `evaluate_model(model, X_test, y_test, model_name="Model")` for detailed metrics.

### Decision Tree Regressor
- Uses GridSearchCV to tune `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- Prints best parameters and evaluation metrics directly.

---

## Getting Started

### Prerequisites
- Python 3.6+
- scikit-learn
- numpy
- pandas (for data handling, if needed)

You can install dependencies with:
```bash
pip install scikit-learn numpy pandas
