"""
Price Prediction Model Training Pipeline

This module provides a complete training pipeline for a price prediction model using LightGBM.
It combines text and image embeddings to predict product prices with log-transformed targets
to handle price variance and uses SMAPE (Symmetric Mean Absolute Percentage Error) for evaluation.

The pipeline includes:
- Loading and validating embedding data and CSV files
- Training/validation splitting with log-transformed targets
- LightGBM model training with early stopping
- Model evaluation using SMAPE metric
- Full dataset retraining and test prediction generation
- Optional model persistence and GPU acceleration

Usage:
    python train_regressor.py [--data-dir DATA_DIR] [--out-dir OUT_DIR] 
                             [--use-gpu] [--model-out MODEL_PATH] [--verbose]

Requirements:
    - Pre-generated text and image embeddings (*.npy files)
    - Train/test CSV files with 'price' and 'sample_id' columns
    - Dependencies: numpy, pandas, scikit-learn, lightgbm, tqdm, joblib
"""

import argparse
import logging
import joblib
import os
import numpy as np
import pandas as pd
from typing import Tuple

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def setup_logger(level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger for the training pipeline.
    
    Sets up a stream handler with timestamp formatting for consistent logging
    throughout the training process. Prevents duplicate handlers on multiple calls.
    
    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        logging.Logger: Configured logger instance for pipeline logging
    """
    logger = logging.getLogger('train_regressor')
    
    # Avoid adding multiple handlers if logger already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


def load_data(data_dir: str, out_dir: str, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load training/test CSVs and pre-computed embeddings from disk.
    
    Loads train.csv and test.csv from data_dir, and corresponding text/image
    embeddings from out_dir. Validates that embedding counts match CSV row counts
    and concatenates text+image embeddings into feature matrices.
    
    Args:
        data_dir (str): Directory containing train.csv and test.csv files
        out_dir (str): Directory containing embedding .npy files
        logger (logging.Logger): Logger for progress and error reporting
        
    Returns:
        Tuple containing:
            - X_train (np.ndarray): Training features (text + image embeddings)
            - X_test (np.ndarray): Test features (text + image embeddings) 
            - train_df (pd.DataFrame): Training CSV data
            - test_df (pd.DataFrame): Test CSV data
            - y_train (np.ndarray): Training target prices
            
    Raises:
        AssertionError: If embedding dimensions don't match CSV row counts
        FileNotFoundError: If required CSV or embedding files are missing
    """
    # Construct file paths
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    logger.info('Loading CSVs from %s and %s', train_csv, test_csv)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Load pre-computed embedding arrays
    logger.info('Loading embeddings from %s', out_dir)
    train_text_emb = np.load(os.path.join(out_dir, 'train_text_emb.npy'))
    train_img_emb = np.load(os.path.join(out_dir, 'train_image_emb.npy'))
    test_text_emb = np.load(os.path.join(out_dir, 'test_text_emb.npy'))
    test_img_emb = np.load(os.path.join(out_dir, 'test_image_emb.npy'))

    # Validate embedding dimensions match CSV row counts
    assert train_text_emb.shape[0] == train_img_emb.shape[0] == len(train_df), 'Train embeddings and CSV length mismatch'
    assert test_text_emb.shape[0] == test_img_emb.shape[0] == len(test_df), 'Test embeddings and CSV length mismatch'

    # Concatenate text and image embeddings along feature axis
    X_train = np.concatenate([train_text_emb, train_img_emb], axis=1)
    X_test = np.concatenate([test_text_emb, test_img_emb], axis=1)
    
    # Extract target prices for training
    y_train = train_df['price'].values

    logger.info('Shapes: X_train=%s, X_test=%s', X_train.shape, X_test.shape)
    return X_train, X_test, train_df, test_df, y_train


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, logger: logging.Logger, use_gpu: bool = False) -> Tuple[lgb.Booster, np.ndarray, np.ndarray]:
    """
    Train a LightGBM regressor with validation and return trained model.
    
    Performs log transformation on target prices, splits data for validation,
    trains LightGBM with early stopping, and evaluates using SMAPE metric.
    The log transformation helps stabilize variance in price predictions.
    
    Args:
        X_train (np.ndarray): Training feature matrix (concatenated embeddings)
        y_train (np.ndarray): Training target prices (original scale)
        logger (logging.Logger): Logger for training progress
        use_gpu (bool): Whether to enable GPU acceleration for LightGBM
        
    Returns:
        Tuple containing:
            - model (lgb.Booster): Trained LightGBM model
            - X_val (np.ndarray): Validation features for further analysis
            - y_val (np.ndarray): Validation targets (log-transformed)
            
    Note:
        Uses log1p transformation on targets to handle price variance.
        SMAPE is computed on original price scale for interpretability.
    """
    # Apply log transformation to stabilize price variance
    y_log = np.log1p(y_train)
    
    # Split into train/validation sets (85%/15%)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_log, test_size=0.15, random_state=42)
    logger.info('Split shapes: train=%s, val=%s', X_tr.shape, X_val.shape)

    # LightGBM hyperparameters optimized for embedding features
    params = {
        'objective': 'regression',        # Regression task
        'metric': 'rmse',                # Root mean squared error
        'learning_rate': 0.08,           # Conservative learning rate
        'num_leaves': 63,                # Tree complexity
        'feature_fraction': 0.7,         # Feature sampling for regularization
        'bagging_fraction': 0.7,         # Row sampling for regularization  
        'bagging_freq': 3,               # Bagging frequency
        'verbose': -1,                   # Suppress verbose output
        'seed': 42                       # Reproducibility
    }
    
    # Add GPU acceleration if requested and available
    if use_gpu:
        params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})

    # Convert to LightGBM datasets (float32 for memory efficiency)
    train_data = lgb.Dataset(X_tr.astype('float32'), label=y_tr)
    val_data = lgb.Dataset(X_val.astype('float32'), label=y_val, reference=train_data)

    # Training configuration
    num_boost_round = 1000  # Maximum iterations
    stopping_rounds = 60    # Early stopping patience

    # Setup progress tracking
    pbar = tqdm(total=num_boost_round, desc='Training Progress', leave=True)

    def progress_callback(env):
        """Update progress bar during training."""
        pbar.update(1)
        if env.iteration + 1 == num_boost_round or env.model is None:
            pbar.close()

    # Train model with validation and early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=num_boost_round,
        callbacks=[
            early_stopping(stopping_rounds=stopping_rounds),
            log_evaluation(50),  # Log every 50 iterations
            progress_callback
        ]
    )

    pbar.close()
    logger.info('Training complete. Best iteration: %s', model.best_iteration)

    # Generate validation predictions and evaluate
    y_val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Convert back to original price scale for evaluation
    y_val_pred = np.expm1(y_val_pred_log)
    y_val_true = np.expm1(y_val)

    def smape(y_true, y_pred):
        """
        Symmetric Mean Absolute Percentage Error.
        
        SMAPE is robust to scale and provides interpretable percentage error.
        """
        eps = 1e-9  # Prevent division by zero
        return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + eps))

    score = smape(y_val_true, y_val_pred)
    logger.info('Validation SMAPE: %.3f%%', score)

    return model, X_val, y_val


def train_full_model_and_predict(model: lgb.Booster, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, logger: logging.Logger) -> Tuple[np.ndarray, lgb.Booster]:
    """
    Retrain model on full dataset and generate test predictions.
    
    Uses the hyperparameters from the validated model but trains on the complete
    training set for maximum performance. Applies the same log transformation
    and converts predictions back to original price scale.
    
    Args:
        model (lgb.Booster): Previously trained model (for hyperparameters)
        X_train (np.ndarray): Complete training feature matrix
        y_train (np.ndarray): Complete training target prices
        X_test (np.ndarray): Test feature matrix for prediction
        logger (logging.Logger): Logger for progress reporting
        
    Returns:
        Tuple containing:
            - test_preds (np.ndarray): Test predictions in original price scale
            - final_model (lgb.Booster): Retrained model on full dataset
            
    Note:
        Uses best_iteration from validation to prevent overfitting.
        Clips predictions to ensure non-negative prices.
    """
    # Apply same log transformation as in validation
    y_log = np.log1p(y_train)
    
    # Create dataset from full training set
    full_data = lgb.Dataset(X_train.astype('float32'), label=y_log)

    # Retrain using same hyperparameters but on full dataset
    final_model = lgb.train(
        model.params,
        full_data,
        num_boost_round=model.best_iteration or 1000  # Use validated stopping point
    )

    # Generate test predictions in log space
    test_preds_log = final_model.predict(X_test.astype('float32'), num_iteration=final_model.best_iteration)
    
    # Convert back to original price scale
    test_preds = np.expm1(test_preds_log)
    
    # Ensure non-negative prices (clip at 0)
    test_preds = np.clip(test_preds, a_min=0, a_max=None)
    
    logger.info('Generated test predictions')
    return test_preds, final_model


def save_model(model: lgb.Booster, model_path: str, logger: logging.Logger) -> None:
    """
    Save trained LightGBM model to disk using joblib.
    
    Persists the complete model including hyperparameters and learned weights
    for later inference or deployment.
    
    Args:
        model (lgb.Booster): Trained LightGBM model to save
        model_path (str): File path where model should be saved
        logger (logging.Logger): Logger for confirmation messaging
        
    Returns:
        None
    """
    joblib.dump(model, model_path)
    logger.info('Saved model to %s', model_path)


def parse_args():
    """
    Parse command-line arguments for the training pipeline.
    
    Returns:
        argparse.Namespace: Parsed arguments with default values
    """
    parser = argparse.ArgumentParser(
        description='Train LightGBM regressor on combined text and image embeddings for price prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data input/output directories
    parser.add_argument(
        '--data-dir', 
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data'),
        help='Directory containing train.csv and test.csv files'
    )
    parser.add_argument(
        '--out-dir', 
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'outputs'),
        help='Directory containing embedding files and where outputs will be saved'
    )
    
    # Training options
    parser.add_argument(
        '--use-gpu', 
        action='store_true', 
        help='Enable LightGBM GPU training (requires GPU-enabled LightGBM installation)'
    )
    
    # Model persistence
    parser.add_argument(
        '--model-out', 
        default=None, 
        help='Path to save trained model using joblib (e.g., model.pkl)'
    )
    
    # Logging control
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def main():
    """
    Main training pipeline entry point.
    
    Orchestrates the complete training workflow:
    1. Parse command-line arguments and setup logging
    2. Load training/test data and embeddings  
    3. Train model with validation and early stopping
    4. Retrain on full dataset for final predictions
    5. Generate submission file and optionally save model
    
    The pipeline uses log-transformed targets and SMAPE evaluation,
    which are appropriate for price prediction tasks with high variance.
    """
    # Parse arguments and configure logging
    args = parse_args()
    logger = setup_logger(logging.DEBUG if args.verbose else logging.INFO)
    
    logger.info('Starting price prediction training pipeline')
    logger.info('Data directory: %s', args.data_dir)
    logger.info('Output directory: %s', args.out_dir)
    logger.info('GPU acceleration: %s', 'enabled' if args.use_gpu else 'disabled')

    # Load and prepare data
    X_train, X_test, train_df, test_df, y_train = load_data(args.data_dir, args.out_dir, logger)

    # Train model with validation for hyperparameter tuning
    model, X_val, y_val = train_and_evaluate(X_train, y_train, logger, use_gpu=args.use_gpu)

    # Retrain on full dataset and generate test predictions
    test_preds, final_model = train_full_model_and_predict(model, X_train, y_train, X_test, logger)

    # Create submission file with predictions
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'], 
        'price': test_preds
    })
    out_path = os.path.join(args.out_dir, 'test_out.csv')
    submission.to_csv(out_path, index=False)
    logger.info('Wrote submission to %s', out_path)

    # Optionally save trained model for later use
    if args.model_out:
        save_model(final_model, args.model_out, logger)
        
    logger.info('Training pipeline completed successfully')


if __name__ == '__main__':
    main()
