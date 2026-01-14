# Binary Classification Project

A professional machine learning project comparing multiple algorithms (Decision Tree, KNN, Perceptron, and Logistic Regression) for binary classification tasks. The project includes stroke prediction and hiring decision datasets with comprehensive MLflow integration, advanced hyperparameter tuning, and thorough evaluation metrics.

## Features

- **Professional Training Pipeline**: Unified, modular codebase with clean separation of concerns
- **MLflow Integration**: Complete experiment tracking, model versioning, and artifact logging
- **Advanced Hyperparameter Tuning**: RandomizedSearchCV with comprehensive parameter grids
- **Stratified Cross-Validation**: Proper 5-fold stratified cross-validation for robust evaluation
- **Class Imbalance Handling**: SMOTE oversampling for imbalanced datasets
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- **Visualization**: Automatic generation of confusion matrices, ROC curves, and PR curves
- **Model Comparison**: Easy comparison of all models across different datasets

## Project Structure

```
Binary classification/
├── code/
│   ├── config.py                 # Hyperparameter configurations
│   ├── data_preprocessing.py      # Data preprocessing module
│   ├── train_pipeline.py          # Main training pipeline with MLflow
│   ├── training_notebook.ipynb    # Interactive training notebook
│   ├── decision_tree_final.ipynb  # Original Decision Tree notebook
│   ├── KNN1005.ipynb              # Original KNN notebook
│   ├── logistic_regression.ipynb  # Original Logistic Regression notebook
│   └── perceptron.ipynb           # Original Perceptron notebook
├── data/
│   ├── stroke/
│   │   └── Dataset1.csv           # Stroke prediction dataset
│   └── HIRING/
│       └── Dataset2.csv           # Hiring decision dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository or navigate to the project directory.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using the Training Pipeline Script

Run the complete training pipeline:

```bash
cd code
python train_pipeline.py
```

This will:
- Preprocess both datasets
- Train all models with hyperparameter tuning
- Log everything to MLflow
- Generate comparison reports

### Option 2: Using the Jupyter Notebook

For interactive exploration:

```bash
cd code
jupyter notebook training_notebook.ipynb
```

### Option 3: Programmatic Usage

```python
from train_pipeline import BinaryClassificationTrainer
from data_preprocessing import DataPreprocessor

# Initialize trainer
trainer = BinaryClassificationTrainer(
    experiment_name="my_experiment",
    tracking_uri="file:./mlruns"
)

# Load and preprocess data
X, y = trainer.preprocessor.preprocess_stroke_data("../data/stroke/Dataset1.csv")
X_train, X_test, y_train, y_test = trainer.preprocessor.split_data(X, y)

# Train a specific model
result = trainer.train_model(
    model_name="logistic_regression",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    dataset_name="stroke",
    n_iter=50
)

# Train all models
trainer.train_all_models(
    dataset_name="stroke",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

## MLflow Integration

### Viewing Experiments

After training, view your experiments:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### What's Tracked

- **Parameters**: All hyperparameters for each model
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, CV scores
- **Artifacts**: 
  - Trained models
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - Feature importance (for tree-based models)

### Experiment Management

- Each model-dataset combination creates a separate run
- All runs are organized under the experiment name
- Easy comparison of different hyperparameter configurations
- Model versioning and reproducibility

## Models and Hyperparameter Tuning

### Decision Tree
- **Parameters tuned**: max_depth, min_samples_split, min_samples_leaf, criterion, class_weight
- **Search space**: Comprehensive grid with 7×4×4×2×2 = 448 combinations (sampled)

### K-Nearest Neighbors (KNN)
- **Parameters tuned**: n_neighbors, weights, metric, p
- **Search space**: 7×2×3×2 = 84 combinations (sampled)

### Logistic Regression
- **Parameters tuned**: C, penalty, solver, class_weight, max_iter
- **Search space**: 6×3×3×2×2 = 216 combinations (sampled)

### Perceptron
- **Parameters tuned**: max_iter, eta0, class_weight, tol, penalty
- **Search space**: 4×4×2×3×4 = 384 combinations (sampled)

All models use **RandomizedSearchCV** with **StratifiedKFold** cross-validation (5 folds) for robust hyperparameter selection.

## Evaluation Metrics

The pipeline calculates and logs:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (handles zero division)
- **Recall**: Recall score (handles zero division)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve
- **CV Mean/Std**: Cross-validation scores with standard deviation

## Class Imbalance Handling

The pipeline automatically:
- Detects class imbalance
- Applies SMOTE (Synthetic Minority Oversampling Technique) to training data
- Uses stratified train-test splitting
- Uses stratified cross-validation

## Configuration

Edit `code/config.py` to customize:
- Hyperparameter grids
- Cross-validation folds
- Scoring metrics
- MLflow settings

## Best Practices Implemented

1. **Reproducibility**: Fixed random seeds throughout
2. **Modularity**: Clean separation of preprocessing, training, and evaluation
3. **Error Handling**: Graceful handling of edge cases
4. **Documentation**: Comprehensive docstrings and comments
5. **Type Hints**: Type annotations for better code clarity
6. **Logging**: Comprehensive MLflow logging for experiment tracking
7. **Validation**: Proper train-test split with stratification
8. **Cross-Validation**: Stratified K-fold for robust evaluation

## Example Output

```
============================================================
Training LOGISTIC_REGRESSION on stroke
============================================================
Performing hyperparameter tuning with 50 iterations...
Best parameters: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'class_weight': 'balanced'}
Best CV score (roc_auc): 0.8234

============================================================
Results for LOGISTIC_REGRESSION on stroke
============================================================
Accuracy:  0.7632
Precision: 0.1700
Recall:    0.7419
F1-Score:  0.2778
ROC-AUC:   0.8234
PR-AUC:    0.2856
CV Score:  0.8234 (+/- 0.0123)
============================================================
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## License

This project is for educational purposes.

## Contributing

Feel free to submit issues or pull requests for improvements!
