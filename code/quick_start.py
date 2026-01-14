"""
Quick start script for training binary classification models.
This is a simplified version for quick experimentation.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_pipeline import BinaryClassificationTrainer

def main():
    """Quick start training script."""
    print("="*60)
    print("BINARY CLASSIFICATION - QUICK START")
    print("="*60)
    
    # Initialize trainer
    trainer = BinaryClassificationTrainer()
    
    # Dataset paths (relative to code directory)
    stroke_data_path = "../data/stroke/Dataset1.csv"
    hiring_data_path = "../data/HIRING/Dataset2.csv"
    
    # Check if files exist
    if not os.path.exists(stroke_data_path):
        print(f"Error: {stroke_data_path} not found!")
        print("Please ensure you're running from the code/ directory")
        return
    
    # Process Stroke Dataset
    print("\n[1/4] Processing Stroke Dataset...")
    X_stroke, y_stroke = trainer.preprocessor.preprocess_stroke_data(stroke_data_path)
    X_train_stroke, X_test_stroke, y_train_stroke, y_test_stroke = trainer.preprocessor.split_data(
        X_stroke, y_stroke, test_size=0.2, apply_smote=True
    )
    print(f"  Training samples: {len(X_train_stroke)}, Test samples: {len(X_test_stroke)}")
    
    # Process Hiring Dataset
    print("\n[2/4] Processing Hiring Dataset...")
    X_hiring, y_hiring = trainer.preprocessor.preprocess_hiring_data(hiring_data_path)
    X_train_hiring, X_test_hiring, y_train_hiring, y_test_hiring = trainer.preprocessor.split_data(
        X_hiring, y_hiring, test_size=0.2, apply_smote=True
    )
    print(f"  Training samples: {len(X_train_hiring)}, Test samples: {len(X_test_hiring)}")
    
    # Train models on Stroke Dataset
    print("\n[3/4] Training models on Stroke Dataset...")
    print("  This may take a few minutes...")
    trainer.train_all_models(
        dataset_name="stroke",
        X_train=X_train_stroke,
        y_train=y_train_stroke,
        X_test=X_test_stroke,
        y_test=y_test_stroke,
        n_iter=30  # Reduced for quick start
    )
    
    # Train models on Hiring Dataset
    print("\n[4/4] Training models on Hiring Dataset...")
    print("  This may take a few minutes...")
    trainer.train_all_models(
        dataset_name="hiring",
        X_train=X_train_hiring,
        y_train=y_train_hiring,
        X_test=X_test_hiring,
        y_test=y_test_hiring,
        n_iter=30  # Reduced for quick start
    )
    
    # Print comparisons
    print("\n" + "="*60)
    print("STROKE DATASET - MODEL COMPARISON")
    print("="*60)
    stroke_comparison = trainer.compare_models("stroke")
    print(stroke_comparison)
    
    print("\n" + "="*60)
    print("HIRING DATASET - MODEL COMPARISON")
    print("="*60)
    hiring_comparison = trainer.compare_models("hiring")
    print(hiring_comparison)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nTo view results in MLflow UI, run:")
    print("  mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()
