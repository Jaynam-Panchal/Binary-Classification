"""
Professional data preprocessing module for binary classification tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles data preprocessing for binary classification tasks."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        
    def preprocess_stroke_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess stroke prediction dataset.
        
        Args:
            data_path: Path to the stroke dataset CSV file
            
        Returns:
            Tuple of (features, target)
        """
        df = pd.read_csv(data_path)
        
        # Encode categorical variables
        categorical_mappings = {
            'gender': ['Female', 'Male'],
            'ever_married': ['No', 'Yes'],
            'Residence_type': ['Rural', 'Urban'],
            'smoking_status': ['never smoked', 'formerly smoked', 'smokes'],
            'work_type': ['Never_worked', 'children', 'Govt_job', 'Self-employed', 'Private']
        }
        
        for col, categories in categorical_mappings.items():
            if col in df.columns:
                df[col] = pd.Categorical(df[col], categories=categories).codes
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        # Standardize numerical features
        numerical_features = ['age', 'avg_glucose_level', 'bmi']
        if all(col in df.columns for col in numerical_features):
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Prepare features and target
        X = df.drop(['id', 'stroke'], axis=1, errors='ignore')
        y = df['stroke']
        
        self.feature_columns = X.columns.tolist()
        self.target_column = 'stroke'
        
        return X, y
    
    def preprocess_hiring_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess hiring decision dataset.
        
        Args:
            data_path: Path to the hiring dataset CSV file
            
        Returns:
            Tuple of (features, target)
        """
        df = pd.read_csv(data_path)
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Standardize numerical features
        numerical_features = ['DistanceFromCompany', 'InterviewScore', 'SkillScore', 'PersonalityScore']
        if all(col in df.columns for col in numerical_features):
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Prepare features and target
        X = df.drop(['HiringDecision'], axis=1, errors='ignore')
        y = df['HiringDecision']
        
        self.feature_columns = X.columns.tolist()
        self.target_column = 'HiringDecision'
        
        return X, y
    
    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        apply_smote: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets with optional SMOTE.
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            apply_smote: Whether to apply SMOTE to training data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        if apply_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def get_class_distribution(self, y: pd.Series) -> dict:
        """Get class distribution statistics."""
        counts = y.value_counts().to_dict()
        total = len(y)
        return {
            'class_counts': counts,
            'class_proportions': {k: v/total for k, v in counts.items()},
            'total_samples': total,
            'is_imbalanced': max(counts.values()) / min(counts.values()) > 2
        }
