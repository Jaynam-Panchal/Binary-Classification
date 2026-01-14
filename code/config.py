"""
Configuration file for hyperparameter tuning and model training.
"""

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

# Cross-validation configuration
CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Hyperparameter grids for each model
HYPERPARAMETER_GRIDS = {
    'decision_tree': {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]  # For minkowski metric
    },
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'lbfgs', 'saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000, 2000]
    },
    'perceptron': {
        'max_iter': [1000, 2000, 3000, 5000],
        'eta0': [0.0001, 0.001, 0.01, 0.1],
        'class_weight': [None, 'balanced'],
        'tol': [1e-4, 1e-3, 1e-2],
        'penalty': [None, 'l2', 'l1', 'elasticnet']
    }
}

# Model classes
MODEL_CLASSES = {
    'decision_tree': DecisionTreeClassifier,
    'knn': KNeighborsClassifier,
    'logistic_regression': LogisticRegression,
    'perceptron': Perceptron
}

# Scoring metrics for hyperparameter tuning
SCORING_METRICS = {
    'primary': 'roc_auc',  # Primary metric for optimization
    'secondary': ['f1', 'precision', 'recall', 'accuracy']
}

# MLflow configuration
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "binary_classification"
