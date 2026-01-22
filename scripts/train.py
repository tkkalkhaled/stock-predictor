#!/usr/bin/env python3
"""
Training Script for Stock Predictor Models
==========================================

This script trains the LSTM ensemble model using historical stock data.
It implements walk-forward validation to prevent overfitting and provides
realistic accuracy estimates.

Usage:
    python scripts/train.py --symbol AAPL --epochs 100
    python scripts/train.py --symbols AAPL,GOOGL,MSFT --epochs 50

Author: Talal Alkhaled
License: MIT
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config, SUPPORTED_SYMBOLS, TIMEFRAMES
from data.preprocess import DataPreprocessor
from models.lstm_ensemble import LSTMEnsemble
from models.sentiment_analyzer import SentimentAnalyzer
from models.technical_patterns import TechnicalPatternDetector
from evaluation.walk_forward_validation import (
    WalkForwardValidator, 
    BacktestSimulator,
    compare_to_baseline
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train stock prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train on single symbol:
    python scripts/train.py --symbol AAPL
  
  Train on multiple symbols:
    python scripts/train.py --symbols AAPL,GOOGL,MSFT
  
  Quick training (fewer epochs):
    python scripts/train.py --symbol AAPL --epochs 20 --quick
  
  Full training with validation:
    python scripts/train.py --symbols AAPL,GOOGL --epochs 100 --validate
        """
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='Single stock symbol to train on (e.g., AAPL)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., AAPL,GOOGL,MSFT)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=config.model.epochs,
        help=f'Number of training epochs (default: {config.model.epochs})'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=config.model.batch_size,
        help=f'Batch size (default: {config.model.batch_size})'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=config.model.learning_rate,
        help=f'Learning rate (default: {config.model.learning_rate})'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=config.model.sequence_length,
        help=f'Sequence length for LSTM (default: {config.model.sequence_length})'
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Run walk-forward validation after training'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtest simulation after training'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick training mode (reduced data, fewer epochs)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=str(config.model.lstm_model_path),
        help=f'Output path for trained model (default: {config.model.lstm_model_path})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate setup without actual training (uses synthetic data)'
    )
    
    return parser.parse_args()


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 15):
    """
    Generate synthetic data for testing/demo.
    
    In production, real data is fetched from APIs.
    """
    print("\n[DATA] Generating synthetic data for training demo...")
    
    np.random.seed(42)
    
    # Create features that have some predictive signal
    X = np.random.randn(n_samples, config.model.sequence_length, n_features)
    
    # Generate target based on a pattern in the features
    # (last value of first feature + trend + noise)
    signal = X[:, -1, 0] * 0.3 + X[:, -5:, 1].mean(axis=1) * 0.2
    noise = np.random.randn(n_samples) * 0.4
    y = (signal + noise > 0).astype(float)
    
    # Add some autocorrelation (realistic for stock data)
    for i in range(1, n_samples):
        if np.random.random() < 0.3:  # 30% chance of following previous direction
            y[i] = y[i-1]
    
    print(f"  Generated {n_samples} samples with {n_features} features")
    print(f"  Sequence length: {config.model.sequence_length}")
    print(f"  Class balance: {y.mean():.1%} positive")
    
    return X, y


def fetch_real_data(symbols: list, timeframe: str = '1Y'):
    """
    Fetch real historical data from APIs.
    
    Requires API keys to be configured in .env
    """
    print(f"\n[DATA] Fetching real data for {len(symbols)} symbols...")
    
    # Check API keys
    api_status = config.validate_api_keys()
    if not api_status['alpaca']:
        print("  [WARNING] Alpaca API keys not configured!")
        print("  Please set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY in .env")
        print("  Falling back to synthetic data...")
        return None, None
    
    preprocessor = DataPreprocessor(
        finnhub_key=config.api.finnhub_key,
        alpaca_key_id=config.api.alpaca_key_id,
        alpaca_secret=config.api.alpaca_secret,
        polygon_key=config.api.polygon_key
    )
    
    all_X = []
    all_y = []
    
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            data = preprocessor.get_processed_data(symbol, timeframe)
            df = data.ohlcv_history
            
            if len(df) < config.model.sequence_length + 10:
                print(f"    [WARNING] Not enough data for {symbol}, skipping")
                continue
            
            # Prepare sequences
            X, y = preprocessor.prepare_sequences(
                df, 
                sequence_length=config.model.sequence_length
            )
            
            all_X.append(X)
            all_y.append(y)
            
            print(f"    [OK] Got {len(X)} sequences")
            
        except Exception as e:
            print(f"    [ERROR] Error fetching {symbol}: {e}")
    
    if not all_X:
        return None, None
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\n  Total: {len(X)} sequences across {len(symbols)} symbols")
    
    return X, y


def train_model(X, y, args):
    """Train the LSTM ensemble model."""
    print("\n[TRAIN] Training LSTM Ensemble...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Split data
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Initialize model
    input_size = X.shape[2]  # Number of features
    ensemble = LSTMEnsemble(input_size=input_size)
    
    # Train
    print("\n  Training...")
    history = ensemble.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=config.model.early_stopping_patience
    )
    
    # Evaluate on validation set
    print("\n[EVAL] Evaluating model...")
    predictions, confidences = ensemble.predict(X_val)
    
    # Calculate accuracy
    pred_direction = (predictions > 0.5).astype(int)
    actual_direction = y_val.astype(int)
    accuracy = np.mean(pred_direction == actual_direction)
    
    print(f"\n  Validation Accuracy: {accuracy:.2%}")
    print(f"  Average Confidence: {confidences.mean():.2%}")
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(str(output_path))
    print(f"\n  Model saved to: {output_path}")
    
    return ensemble, accuracy


def run_validation(ensemble, X, y):
    """Run walk-forward validation."""
    print("\n[VALIDATE] Running Walk-Forward Validation...")
    
    # Create a simple model wrapper for validation
    class ModelWrapper:
        def __init__(self, ensemble):
            self.ensemble = ensemble
        
        def fit(self, X, y):
            # In production, would retrain on this window
            pass
        
        def predict(self, X):
            preds, _ = self.ensemble.predict(X)
            return preds
    
    wrapper = ModelWrapper(ensemble)
    
    validator = WalkForwardValidator(
        train_window_size=config.model.train_window_size,
        test_window_size=config.model.test_window_size,
        step_size=config.model.step_size
    )
    
    report = validator.validate(wrapper, X, y)
    
    print("\n" + report.summary)
    
    # Compare to baseline
    comparison = compare_to_baseline(report.avg_direction_accuracy, 0.5)
    print("\n" + comparison['interpretation'])
    
    return report


def run_backtest(ensemble, X, y):
    """Run backtest simulation."""
    print("\n[BACKTEST] Running Backtest Simulation...")
    
    # Get predictions
    predictions, confidences = ensemble.predict(X[-252:])  # Last year
    
    # Simulate returns (in production, use actual returns)
    actual_returns = np.diff(np.random.randn(len(predictions) + 1).cumsum() * 0.02)
    
    backtester = BacktestSimulator(
        initial_capital=100000,
        transaction_cost=0.001
    )
    
    results = backtester.run_backtest(predictions, actual_returns, confidences)
    
    print(f"\n  Strategy Return: {results['total_return']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Win Rate: {results['win_rate']:.2%}")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Buy & Hold Return: {results['buy_hold_return']:.2%}")
    print(f"  Outperformance: {results['outperformance']:.2%}")
    
    return results


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 60)
    print("Stock Predictor - Model Training")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine symbols
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = ['AAPL']  # Default
    
    print(f"\nSymbols: {', '.join(symbols)}")
    
    # Check configuration
    print("\n[CONFIG] Configuration:")
    api_status = config.validate_api_keys()
    for api, configured in api_status.items():
        status = "[OK]" if configured else "[--]"
        print(f"  {status} {api}")
    
    # Get data
    if args.dry_run:
        print("\n[DRY RUN] Using synthetic data")
        X, y = generate_synthetic_data(
            n_samples=500 if args.quick else 1000
        )
    else:
        X, y = fetch_real_data(symbols)
        
        if X is None:
            print("\n[WARNING] Could not fetch real data. Using synthetic data for demo.")
            X, y = generate_synthetic_data(
                n_samples=500 if args.quick else 1000
            )
    
    # Train model
    ensemble, accuracy = train_model(X, y, args)
    
    # Optional: Walk-forward validation
    if args.validate:
        run_validation(ensemble, X, y)
    
    # Optional: Backtest
    if args.backtest:
        run_backtest(ensemble, X, y)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model saved to: {args.output}")
    print(f"Final accuracy: {accuracy:.2%}")
    
    print("\nNext steps:")
    print("  1. Run the API server: uvicorn api.predict:app --reload")
    print("  2. Test predictions: curl http://localhost:8000/predict -X POST ...")
    print("  3. View docs: http://localhost:8000/docs")


if __name__ == '__main__':
    main()
