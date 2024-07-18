from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging
import sys
import os

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATASET_MAP = {
    "361092": "y_prop",
    "361098": "brazilian_houses",
    "361099": "bike_sharing"
}

def wrangle_data(X: pd.DataFrame) -> pd.DataFrame:
    # Identify binary and non-binary categorical columns
    binary_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns if X[col].nunique() == 2]
    non_binary_cols = [col for col in X.select_dtypes(include=['object', 'category']).columns if X[col].nunique() > 2]
    
    # Convert binary categorical data to 0/1 in a single column
    for col in binary_cols:
        X[col] = X[col].astype('category').cat.codes
    
    # One-hot encode non-binary categorical data
    X = pd.get_dummies(X, columns=non_binary_cols, drop_first=True)
    
    # Feature scaling (standardization)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled

def load_all_folds(task: str, input_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    all_X = []
    all_y = []
    task_dir = input_dir / task
    folds = [fold for fold in task_dir.iterdir() if fold.is_dir()]
    
    for fold in folds:
        X_train_path = fold / "X_train.parquet"
        y_train_path = fold / "y_train.parquet"
        X_test_path = fold / "X_test.parquet"
        y_test_path = fold / "y_test.parquet"
        
        if X_train_path.exists() and y_train_path.exists():
            X_train = pd.read_parquet(X_train_path)
            y_train = pd.read_parquet(y_train_path).iloc[:, 0]
            all_X.append(X_train)
            all_y.append(y_train)
        
        if X_test_path.exists() and y_test_path.exists():
            X_test = pd.read_parquet(X_test_path)
            y_test = pd.read_parquet(y_test_path).iloc[:, 0]
            all_X.append(X_test)
            all_y.append(y_test)
    
    X_full = pd.concat(all_X, axis=0)
    y_full = pd.concat(all_y, axis=0)
    
    return X_full, y_full

def process_and_save_data(task: str, input_dir: Path, output_dir: Path) -> None:
    try:
        # Load all data from all folds
        X_full, y_full = load_all_folds(task, input_dir)
        logger.info(f"Loaded full dataset for task '{task}'")

        # Wrangle the full dataset
        X_full_wrangled = wrangle_data(X_full)
        
        # Split back into original folds
        task_dir = input_dir / task
        folds = [fold for fold in task_dir.iterdir() if fold.is_dir()]
        
        start = 0
        for fold in folds:
            X_train_path = fold / "X_train.parquet"
            y_train_path = fold / "y_train.parquet"
            X_test_path = fold / "X_test.parquet"
            y_test_path = fold / "y_test.parquet"

            fold_dir = output_dir / task / fold.name
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            if X_train_path.exists() and y_train_path.exists():
                X_train = pd.read_parquet(X_train_path)
                y_train = pd.read_parquet(y_train_path).iloc[:, 0]
                end = start + len(X_train)
                X_train_wrangled = X_full_wrangled.iloc[start:end, :]
                start = end
                X_train_wrangled.to_parquet(fold_dir / "X_train.parquet")
                y_train.to_frame().to_parquet(fold_dir / "y_train.parquet")
                
            if X_test_path.exists() and y_test_path.exists():
                X_test = pd.read_parquet(X_test_path)
                y_test = pd.read_parquet(y_test_path).iloc[:, 0]
                end = start + len(X_test)
                X_test_wrangled = X_full_wrangled.iloc[start:end, :]
                start = end
                X_test_wrangled.to_parquet(fold_dir / "X_test.parquet")
                y_test.to_frame().to_parquet(fold_dir / "y_test.parquet")
        
        logger.info(f"Wrangled data for task '{task}'")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def main(input_dir: Path, output_dir: Path, tasks: list[str]) -> None:
    for task in tasks:
        process_and_save_data(task, input_dir, output_dir)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Use os.path to correctly handle paths on Windows
    input_dir = Path(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'external')).resolve()
    output_dir = Path(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')).resolve()

    parser.add_argument("--input-dir", type=Path, default=input_dir, help="The directory where the external datasets are stored.")
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="The directory where the processed datasets will be stored.")
    parser.add_argument("--tasks", nargs='+', default=["361092", "361098", "361099"], help="The list of tasks to wrangle data for.")
    
    args = parser.parse_args()
    
    main(input_dir=args.input_dir, output_dir=args.output_dir, tasks=args.tasks)