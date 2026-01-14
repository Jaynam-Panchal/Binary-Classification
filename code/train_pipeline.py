"""
Professional training pipeline with MLflow integration for binary classification.
"""

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import joblib
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

try:
    from .data_preprocessing import DataPreprocessor
    from .config import (
        HYPERPARAMETER_GRIDS, MODEL_CLASSES, SCORING_METRICS,
        CV_FOLDS, RANDOM_STATE, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    )
except ImportError:
    from data_preprocessing import DataPreprocessor
    from config import (
        HYPERPARAMETER_GRIDS, MODEL_CLASSES, SCORING_METRICS,
        CV_FOLDS, RANDOM_STATE, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
    )


class BinaryClassificationTrainer:
    """Professional trainer for binary classification models with MLflow integration."""
    
    def __init__(
        self,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str = MLFLOW_TRACKING_URI
    ):
        """
        Initialize the trainer.
        
        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.preprocessor = DataPreprocessor(random_state=RANDOM_STATE)
        self.models = {}
        self.results = {}
        
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_name: str,
        n_iter: int = 50,
        cv: int = CV_FOLDS
    ) -> Dict[str, Any]:
        """
        Train a model with hyperparameter tuning and MLflow logging.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            dataset_name: Name of the dataset
            n_iter: Number of iterations for RandomizedSearchCV
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary containing model and metrics
        """
        if model_name not in MODEL_CLASSES:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()} on {dataset_name}")
        print(f"{'='*60}")
        
        # Get model class and hyperparameter grid
        model_class = MODEL_CLASSES[model_name]
        param_grid = HYPERPARAMETER_GRIDS[model_name]
        
        # Create base model
        base_model = model_class(random_state=RANDOM_STATE)
        
        # Setup cross-validation
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        
        # Hyperparameter tuning with RandomizedSearchCV
        print(f"Performing hyperparameter tuning with {n_iter} iterations...")
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=SCORING_METRICS['primary'],
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score ({SCORING_METRICS['primary']}): {random_search.best_score_:.4f}")
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        # Perceptron doesn't have predict_proba, use decision_function if available
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, 'decision_function'):
            # Convert decision function to probability-like scores using sigmoid
            decision_scores = best_model.decision_function(X_test)
            # Normalize to [0, 1] range using sigmoid
            y_pred_proba = expit(decision_scores)
        else:
            y_pred_proba = None
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Cross-validation scores on training data
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=cv_strategy, scoring=SCORING_METRICS['primary']
        )
        
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
            # Log parameters
            mlflow.log_params(random_search.best_params_)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("cv_folds", cv)
            mlflow.log_param("n_iter", n_iter)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Log artifacts (plots)
            self._log_artifacts(y_test, y_pred, y_pred_proba, model_name, dataset_name)
            
            # Log feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                mlflow.log_table(data=feature_importance, artifact_file="feature_importance.json")
        
        # Store results
        result = {
            'model': best_model,
            'best_params': random_search.best_params_,
            'metrics': metrics,
            'cv_scores': cv_scores,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba
        }
        
        self.models[f"{model_name}_{dataset_name}"] = best_model
        self.results[f"{model_name}_{dataset_name}"] = result
        
        # Print summary
        self._print_summary(metrics, model_name, dataset_name)
        
        return result
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    def _log_artifacts(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        model_name: str,
        dataset_name: str
    ):
        """Create and log visualization artifacts."""
        # Create output directory
        os.makedirs("mlflow_artifacts", exist_ok=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"mlflow_artifacts/confusion_matrix_{model_name}_{dataset_name}.png")
        plt.close()
        mlflow.log_artifact(f"mlflow_artifacts/confusion_matrix_{model_name}_{dataset_name}.png")
        
        # ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label='ROC Curve')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name} ({dataset_name})')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"mlflow_artifacts/roc_curve_{model_name}_{dataset_name}.png")
            plt.close()
            mlflow.log_artifact(f"mlflow_artifacts/roc_curve_{model_name}_{dataset_name}.png")
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name} ({dataset_name})')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"mlflow_artifacts/pr_curve_{model_name}_{dataset_name}.png")
            plt.close()
            mlflow.log_artifact(f"mlflow_artifacts/pr_curve_{model_name}_{dataset_name}.png")
    
    def _print_summary(self, metrics: Dict[str, float], model_name: str, dataset_name: str):
        """Print training summary."""
        print(f"\n{'='*60}")
        print(f"Results for {model_name.upper()} on {dataset_name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
        if 'cv_mean' in metrics:
            print(f"CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        print(f"{'='*60}\n")
    
    def train_all_models(
        self,
        dataset_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models: Optional[list] = None,
        n_iter: int = 50
    ):
        """
        Train all models on a dataset.
        
        Args:
            dataset_name: Name of the dataset
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            models: List of model names to train (None for all)
            n_iter: Number of iterations for hyperparameter tuning
        """
        if models is None:
            models = list(MODEL_CLASSES.keys())
        
        for model_name in models:
            try:
                self.train_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    dataset_name=dataset_name,
                    n_iter=n_iter
                )
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
    
    def compare_models(self, dataset_name: str) -> pd.DataFrame:
        """
        Compare all trained models for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for key, result in self.results.items():
            if dataset_name in key:
                model_name = key.replace(f"_{dataset_name}", "")
                metrics = result['metrics']
                metrics['model'] = model_name
                comparison_data.append(metrics)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.set_index('model')
            return df.sort_values('roc_auc' if 'roc_auc' in df.columns else 'f1_score', ascending=False)
        else:
            return pd.DataFrame()
    
    def save_model(self, model_key: str, filepath: str):
        """Save a trained model to disk."""
        if model_key in self.models:
            joblib.dump(self.models[model_key], filepath)
            print(f"Model saved to {filepath}")
        else:
            raise ValueError(f"Model {model_key} not found")


def main():
    """Main training pipeline."""
    # Initialize trainer
    trainer = BinaryClassificationTrainer()
    
    # Dataset paths
    stroke_data_path = "../data/stroke/Dataset1.csv"
    hiring_data_path = "../data/HIRING/Dataset2.csv"
    
    # Process Stroke Dataset
    print("\n" + "="*60)
    print("PROCESSING STROKE DATASET")
    print("="*60)
    X_stroke, y_stroke = trainer.preprocessor.preprocess_stroke_data(stroke_data_path)
    X_train_stroke, X_test_stroke, y_train_stroke, y_test_stroke = trainer.preprocessor.split_data(
        X_stroke, y_stroke, test_size=0.2, apply_smote=True
    )
    
    stroke_dist = trainer.preprocessor.get_class_distribution(y_stroke)
    print(f"Class distribution: {stroke_dist['class_counts']}")
    print(f"Is imbalanced: {stroke_dist['is_imbalanced']}")
    
    # Process Hiring Dataset
    print("\n" + "="*60)
    print("PROCESSING HIRING DATASET")
    print("="*60)
    X_hiring, y_hiring = trainer.preprocessor.preprocess_hiring_data(hiring_data_path)
    X_train_hiring, X_test_hiring, y_train_hiring, y_test_hiring = trainer.preprocessor.split_data(
        X_hiring, y_hiring, test_size=0.2, apply_smote=True
    )
    
    hiring_dist = trainer.preprocessor.get_class_distribution(y_hiring)
    print(f"Class distribution: {hiring_dist['class_counts']}")
    print(f"Is imbalanced: {hiring_dist['is_imbalanced']}")
    
    # Train all models on Stroke Dataset
    print("\n" + "="*60)
    print("TRAINING MODELS ON STROKE DATASET")
    print("="*60)
    trainer.train_all_models(
        dataset_name="stroke",
        X_train=X_train_stroke,
        y_train=y_train_stroke,
        X_test=X_test_stroke,
        y_test=y_test_stroke,
        n_iter=50
    )
    
    # Train all models on Hiring Dataset
    print("\n" + "="*60)
    print("TRAINING MODELS ON HIRING DATASET")
    print("="*60)
    trainer.train_all_models(
        dataset_name="hiring",
        X_train=X_train_hiring,
        y_train=y_train_hiring,
        X_test=X_test_hiring,
        y_test=y_test_hiring,
        n_iter=50
    )
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON - STROKE DATASET")
    print("="*60)
    stroke_comparison = trainer.compare_models("stroke")
    print(stroke_comparison)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON - HIRING DATASET")
    print("="*60)
    hiring_comparison = trainer.compare_models("hiring")
    print(hiring_comparison)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"MLflow UI: Run 'mlflow ui' to view results")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
