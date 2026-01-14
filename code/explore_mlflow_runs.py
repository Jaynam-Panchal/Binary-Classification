"""
Script to explore and analyze MLflow runs programmatically.
Useful for getting insights from your experiments.
"""

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import os

def explore_mlflow_runs(experiment_name="binary_classification", tracking_uri="file:./mlruns"):
    """
    Explore MLflow runs and display summary statistics.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: MLflow tracking URI
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found!")
            print("\nAvailable experiments:")
            client = MlflowClient()
            for exp in client.search_experiments():
                print(f"  - {exp.name} (ID: {exp.experiment_id})")
            return
        
        experiment_id = experiment.experiment_id
        print(f"\n{'='*70}")
        print(f"EXPLORING EXPERIMENT: {experiment_name}")
        print(f"Experiment ID: {experiment_id}")
        print(f"{'='*70}\n")
        
        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.roc_auc DESC"]
        )
        
        if runs.empty:
            print("No runs found in this experiment!")
            return
        
        print(f"Total Runs: {len(runs)}\n")
        
        # Display summary by dataset
        print("\n" + "="*70)
        print("SUMMARY BY DATASET")
        print("="*70)
        
        for dataset in runs['params.dataset'].unique():
            dataset_runs = runs[runs['params.dataset'] == dataset]
            print(f"\n📊 {dataset.upper()} DATASET ({len(dataset_runs)} runs)")
            print("-" * 70)
            
            # Best model
            best_run = dataset_runs.iloc[0]
            print(f"\n🏆 Best Model: {best_run['params.model_name']}")
            print(f"   ROC-AUC: {best_run['metrics.roc_auc']:.4f}")
            print(f"   F1-Score: {best_run['metrics.f1_score']:.4f}")
            print(f"   Accuracy: {best_run['metrics.accuracy']:.4f}")
            
            # Model comparison table
            comparison = dataset_runs.groupby('params.model_name').agg({
                'metrics.roc_auc': ['mean', 'max', 'std'],
                'metrics.f1_score': ['mean', 'max'],
                'metrics.accuracy': ['mean', 'max']
            }).round(4)
            
            print(f"\n📈 Model Comparison for {dataset}:")
            print(comparison)
            
            # Best hyperparameters for top model
            print(f"\n🔧 Best Hyperparameters ({best_run['params.model_name']}):")
            best_params = {k.replace('params.', ''): v 
                          for k, v in best_run.items() 
                          if k.startswith('params.') and 
                          k not in ['params.model_name', 'params.dataset', 'params.cv_folds', 'params.n_iter']}
            for param, value in best_params.items():
                print(f"   {param}: {value}")
        
        # Overall statistics
        print("\n" + "="*70)
        print("OVERALL STATISTICS")
        print("="*70)
        
        print(f"\nAverage ROC-AUC: {runs['metrics.roc_auc'].mean():.4f} ± {runs['metrics.roc_auc'].std():.4f}")
        print(f"Best ROC-AUC: {runs['metrics.roc_auc'].max():.4f}")
        print(f"Worst ROC-AUC: {runs['metrics.roc_auc'].min():.4f}")
        
        # Model performance summary
        print("\n" + "="*70)
        print("MODEL PERFORMANCE RANKING")
        print("="*70)
        
        model_summary = runs.groupby('params.model_name').agg({
            'metrics.roc_auc': ['mean', 'max', 'count']
        }).round(4)
        model_summary.columns = ['Mean ROC-AUC', 'Max ROC-AUC', 'Runs']
        model_summary = model_summary.sort_values('Mean ROC-AUC', ascending=False)
        print(model_summary)
        
        # Detailed run information
        print("\n" + "="*70)
        print("DETAILED RUN INFORMATION")
        print("="*70)
        
        display_columns = [
            'run_id',
            'params.model_name',
            'params.dataset',
            'metrics.roc_auc',
            'metrics.f1_score',
            'metrics.accuracy',
            'metrics.cv_mean'
        ]
        
        available_columns = [col for col in display_columns if col in runs.columns]
        detailed_view = runs[available_columns].copy()
        detailed_view.columns = [col.replace('params.', '').replace('metrics.', '') 
                                for col in detailed_view.columns]
        
        print("\nTop 10 Runs (by ROC-AUC):")
        print(detailed_view.head(10).to_string(index=False))
        
        # Save to CSV
        output_file = f"mlflow_runs_summary_{experiment_name}.csv"
        runs[available_columns].to_csv(output_file, index=False)
        print(f"\n✅ Detailed results saved to: {output_file}")
        
        return runs
        
    except Exception as e:
        print(f"Error exploring runs: {str(e)}")
        import traceback
        traceback.print_exc()


def compare_models(runs, dataset_name=None, top_n=5):
    """
    Compare top N models for a dataset.
    
    Args:
        runs: DataFrame of MLflow runs
        dataset_name: Filter by dataset (None for all)
        top_n: Number of top models to compare
    """
    if dataset_name:
        runs = runs[runs['params.dataset'] == dataset_name]
    
    top_runs = runs.head(top_n)
    
    print(f"\n{'='*70}")
    print(f"TOP {top_n} MODELS" + (f" ({dataset_name})" if dataset_name else ""))
    print(f"{'='*70}\n")
    
    for idx, (_, run) in enumerate(top_runs.iterrows(), 1):
        print(f"{idx}. {run['params.model_name']} on {run['params.dataset']}")
        print(f"   ROC-AUC: {run['metrics.roc_auc']:.4f}")
        print(f"   F1-Score: {run['metrics.f1_score']:.4f}")
        print(f"   Accuracy: {run['metrics.accuracy']:.4f}")
        print(f"   Run ID: {run['run_id'][:8]}...")
        print()


def main():
    """Main function."""
    import sys
    
    # Check if experiment name provided
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "binary_classification"
    
    print("MLflow Run Explorer")
    print("="*70)
    
    runs = explore_mlflow_runs(experiment_name=experiment_name)
    
    if runs is not None and not runs.empty:
        # Compare top models for each dataset
        for dataset in runs['params.dataset'].unique():
            compare_models(runs, dataset_name=dataset, top_n=3)
        
        print("\n" + "="*70)
        print("💡 TIP: Use MLflow UI for interactive exploration:")
        print("   mlflow ui")
        print("   Then open http://localhost:5000")
        print("="*70)


if __name__ == "__main__":
    main()
