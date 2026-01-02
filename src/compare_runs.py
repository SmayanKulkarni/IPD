"""
Utility script for comparing MLflow runs across experiments.
Provides easy comparison of hyperparameters, metrics, and results.
"""

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from typing import List, Optional
import argparse


def compare_experiments(experiment_names: List[str] = ["Pose_LSTM_Experiment", "Hybrid_TCN_Experiment"],
                        top_n: Optional[int] = None,
                        metric: str = "best_val_accuracy") -> pd.DataFrame:
    """
    Compare all runs across experiments.
    
    Args:
        experiment_names: List of experiment names to compare
        top_n: If specified, only show top N runs
        metric: Metric to sort by (default: best_val_accuracy)
    
    Returns:
        DataFrame with comparison results
    """
    client = MlflowClient()
    
    all_runs = []
    for exp_name in experiment_names:
        try:
            exp = client.get_experiment_by_name(exp_name)
            if exp:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=[f"metrics.{metric} DESC"]
                )
                all_runs.extend(runs)
        except Exception as e:
            print(f"Warning: Could not fetch runs from '{exp_name}': {e}")
    
    # Create comparison dataframe
    data = []
    for run in all_runs:
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags
        
        data.append({
            'Run Name': tags.get('mlflow.runName', 'unnamed'),
            'Experiment': tags.get('mlflow.experiment.name', 'unknown'),
            'Best Val Acc': metrics.get('best_val_accuracy', 0),
            'Best Val Loss': metrics.get('best_val_loss', 'N/A'),
            'Epochs': params.get('epochs', 'N/A'),
            'Batch Size': params.get('batch_size', 'N/A'),
            'Learning Rate': params.get('learning_rate', 'default'),
            'Train Samples': params.get('dataset.train_samples', 'N/A'),
            'Status': run.info.status,
            'Start Time': pd.to_datetime(run.info.start_time, unit='ms'),
            'Duration (min)': round((run.info.end_time - run.info.start_time) / 60000, 2) if run.info.end_time else 'Running',
            'Git Branch': tags.get('git_branch', 'unknown'),
            'Git Commit': tags.get('git_commit', 'unknown'),
            'Description': tags.get('run_description', ''),
        })
    
    df = pd.DataFrame(data)
    
    # Sort by metric
    if not df.empty and 'Best Val Acc' in df.columns:
        df = df.sort_values('Best Val Acc', ascending=False)
    
    # Limit to top N if specified
    if top_n:
        df = df.head(top_n)
    
    print("\n" + "="*120)
    print("📊 EXPERIMENT COMPARISON")
    print("="*120)
    
    if df.empty:
        print("No runs found.")
    else:
        # Display summary
        print(f"\nTotal runs: {len(df)}")
        print(f"Experiments: {', '.join(experiment_names)}")
        print(f"\nTop runs by {metric}:")
        print("-"*120)
        
        # Format display
        display_cols = ['Run Name', 'Experiment', 'Best Val Acc', 'Best Val Loss', 
                       'Epochs', 'Batch Size', 'Status', 'Duration (min)']
        display_df = df[display_cols].copy()
        
        # Format numerical columns
        if 'Best Val Acc' in display_df.columns:
            display_df['Best Val Acc'] = display_df['Best Val Acc'].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
            )
        if 'Best Val Loss' in display_df.columns:
            display_df['Best Val Loss'] = display_df['Best Val Loss'].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
            )
        
        print(display_df.to_string(index=False))
        print("-"*120)
        
        # Statistics
        print(f"\n📈 STATISTICS:")
        print(f"   Best accuracy: {df['Best Val Acc'].max():.4f}")
        print(f"   Mean accuracy: {df['Best Val Acc'].mean():.4f}")
        print(f"   Std accuracy: {df['Best Val Acc'].std():.4f}")
        print(f"   Completed runs: {(df['Status'] == 'FINISHED').sum()}")
        print(f"   Failed runs: {(df['Status'] == 'FAILED').sum()}")
    
    print("="*120 + "\n")
    
    return df


def compare_specific_runs(run_ids: List[str]) -> pd.DataFrame:
    """
    Compare specific runs by their run IDs.
    
    Args:
        run_ids: List of MLflow run IDs to compare
    
    Returns:
        DataFrame with detailed comparison
    """
    client = MlflowClient()
    
    data = []
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            tags = run.data.tags
            
            row = {
                'Run ID': run_id[:8],
                'Run Name': tags.get('mlflow.runName', 'unnamed'),
            }
            
            # Add all metrics
            for key, value in metrics.items():
                row[f"metric.{key}"] = value
            
            # Add key params
            for key, value in params.items():
                row[f"param.{key}"] = value
            
            data.append(row)
        except Exception as e:
            print(f"Warning: Could not fetch run {run_id}: {e}")
    
    df = pd.DataFrame(data)
    
    print("\n📊 DETAILED RUN COMPARISON")
    print("="*120)
    print(df.to_string(index=False))
    print("="*120 + "\n")
    
    return df


def get_best_run(experiment_name: str, metric: str = "best_val_accuracy") -> dict:
    """
    Get the best run from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize for
    
    Returns:
        Dictionary with best run information
    """
    client = MlflowClient()
    
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            print(f"Experiment '{experiment_name}' not found.")
            return {}
        
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        
        if not runs:
            print(f"No runs found in experiment '{experiment_name}'.")
            return {}
        
        best_run = runs[0]
        
        print("\n🏆 BEST RUN")
        print("="*70)
        print(f"Experiment: {experiment_name}")
        print(f"Run Name: {best_run.data.tags.get('mlflow.runName', 'unnamed')}")
        print(f"Run ID: {best_run.info.run_id}")
        print(f"Best {metric}: {best_run.data.metrics.get(metric, 'N/A')}")
        print(f"Status: {best_run.info.status}")
        print(f"Description: {best_run.data.tags.get('run_description', 'N/A')}")
        print("="*70 + "\n")
        
        return {
            'run_id': best_run.info.run_id,
            'run_name': best_run.data.tags.get('mlflow.runName', 'unnamed'),
            'metric_value': best_run.data.metrics.get(metric, 0),
            'params': best_run.data.params,
            'metrics': best_run.data.metrics,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {}


def main():
    """Command-line interface for run comparison."""
    parser = argparse.ArgumentParser(description="Compare MLflow runs")
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=["Pose_LSTM_Experiment", "Hybrid_TCN_Experiment"],
        help="Experiment names to compare"
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help="Show only top N runs"
    )
    parser.add_argument(
        '--metric',
        default='best_val_accuracy',
        help="Metric to sort by"
    )
    parser.add_argument(
        '--best',
        action='store_true',
        help="Show only the best run from each experiment"
    )
    
    args = parser.parse_args()
    
    if args.best:
        for exp_name in args.experiments:
            get_best_run(exp_name, args.metric)
    else:
        compare_experiments(args.experiments, args.top_n, args.metric)


if __name__ == "__main__":
    main()
