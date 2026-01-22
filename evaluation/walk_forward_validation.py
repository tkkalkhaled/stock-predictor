"""
Walk-Forward Validation for Time Series Models
===============================================

This module implements walk-forward (rolling window) cross-validation,
which is the gold standard for validating time series prediction models.

Unlike traditional k-fold CV, walk-forward validation respects the
temporal ordering of data, preventing look-ahead bias that would
artificially inflate accuracy metrics.

Key Concepts:
- Training window: Fixed or expanding historical data
- Test window: Forward-looking period to evaluate predictions
- Roll: Move both windows forward by a step size

Author: Talal Alkhaled
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class ValidationResult:
    """Container for validation results of a single window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    predictions: np.ndarray
    actuals: np.ndarray
    accuracy: float
    direction_accuracy: float
    mse: float
    mae: float
    sharpe_ratio: float


@dataclass
class FullValidationReport:
    """Complete validation report across all windows."""
    window_results: List[ValidationResult]
    avg_accuracy: float
    avg_direction_accuracy: float
    avg_mse: float
    avg_mae: float
    avg_sharpe: float
    std_accuracy: float
    best_window: int
    worst_window: int
    total_predictions: int
    summary: str


class WalkForwardValidator:
    """
    Walk-forward (rolling window) cross-validation for time series.
    
    This validation approach:
    1. Trains on a window of historical data
    2. Tests on the immediately following period
    3. Rolls both windows forward
    4. Repeats until all data is covered
    
    This prevents data leakage and provides a realistic estimate
    of out-of-sample model performance.
    """
    
    def __init__(
        self,
        train_window_size: int = 252,  # 1 year of trading days
        test_window_size: int = 21,     # 1 month of trading days
        step_size: int = 21,            # Roll forward by 1 month
        expanding_window: bool = False  # If True, train window grows
    ):
        """
        Initialize the validator.
        
        Args:
            train_window_size: Number of periods for training
            test_window_size: Number of periods for testing
            step_size: How many periods to roll forward each iteration
            expanding_window: If True, training window expands over time
        """
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
        self.step_size = step_size
        self.expanding_window = expanding_window
    
    def get_splits(
        self,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index splits for walk-forward validation.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        
        # Start position for training
        train_start = 0
        
        while True:
            # Calculate train end (exclusive) and test range
            if self.expanding_window:
                train_end = train_start + self.train_window_size + \
                           (len(splits) * self.step_size)
            else:
                train_start = len(splits) * self.step_size
                train_end = train_start + self.train_window_size
            
            test_start = train_end
            test_end = test_start + self.test_window_size
            
            # Stop if we've run out of test data
            if test_end > n_samples:
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def validate(
        self,
        model: object,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[np.ndarray] = None,
        fit_method: str = 'fit',
        predict_method: str = 'predict'
    ) -> FullValidationReport:
        """
        Run walk-forward validation on a model.
        
        Args:
            model: Model object with fit and predict methods
            X: Feature matrix (samples x features) or (samples x seq_len x features)
            y: Target values
            dates: Optional array of dates for reporting
            fit_method: Name of the model's training method
            predict_method: Name of the model's prediction method
            
        Returns:
            FullValidationReport with all metrics
        """
        n_samples = len(y)
        splits = self.get_splits(n_samples)
        
        if not splits:
            raise ValueError(
                f"Not enough data for validation. Need at least "
                f"{self.train_window_size + self.test_window_size} samples, "
                f"got {n_samples}"
            )
        
        results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"  Window {i + 1}/{len(splits)}: "
                  f"Train {train_idx[0]}-{train_idx[-1]}, "
                  f"Test {test_idx[0]}-{test_idx[-1]}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            fit_fn = getattr(model, fit_method)
            fit_fn(X_train, y_train)
            
            # Predict
            predict_fn = getattr(model, predict_method)
            predictions = predict_fn(X_test)
            
            # Ensure predictions are numpy array
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            predictions = np.array(predictions).flatten()
            
            # Calculate metrics
            result = self._calculate_metrics(
                predictions=predictions,
                actuals=y_test,
                train_idx=train_idx,
                test_idx=test_idx,
                dates=dates
            )
            results.append(result)
        
        # Aggregate results
        return self._create_report(results)
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        dates: Optional[np.ndarray] = None
    ) -> ValidationResult:
        """Calculate performance metrics for a validation window."""
        
        # For binary classification (direction prediction)
        pred_direction = (predictions > 0.5).astype(int)
        actual_direction = actuals.astype(int)
        
        # Accuracy metrics
        direction_accuracy = np.mean(pred_direction == actual_direction)
        
        # For regression (if predictions are price changes)
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Simple accuracy (within 1% threshold for regression)
        if np.max(actuals) > 1:  # Regression case
            accuracy = np.mean(np.abs(predictions - actuals) / (np.abs(actuals) + 1e-8) < 0.01)
        else:  # Classification case
            accuracy = direction_accuracy
        
        # Sharpe ratio (if we have returns)
        if np.max(actuals) <= 1 and np.min(actuals) >= -1:
            # Actuals are returns/directions
            # Simulate returns based on predictions
            simulated_returns = actuals * (2 * pred_direction - 1)  # Long if pred=1, short if pred=0
            if np.std(simulated_returns) > 0:
                sharpe = np.mean(simulated_returns) / np.std(simulated_returns) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Date ranges
        if dates is not None:
            train_start = dates[train_idx[0]]
            train_end = dates[train_idx[-1]]
            test_start = dates[test_idx[0]]
            test_end = dates[test_idx[-1]]
        else:
            # Use index as pseudo-dates
            train_start = datetime.now() - timedelta(days=len(train_idx) + len(test_idx))
            train_end = datetime.now() - timedelta(days=len(test_idx))
            test_start = train_end
            test_end = datetime.now()
        
        return ValidationResult(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            predictions=predictions,
            actuals=actuals,
            accuracy=accuracy,
            direction_accuracy=direction_accuracy,
            mse=mse,
            mae=mae,
            sharpe_ratio=sharpe
        )
    
    def _create_report(
        self,
        results: List[ValidationResult]
    ) -> FullValidationReport:
        """Create aggregate report from window results."""
        
        accuracies = [r.direction_accuracy for r in results]
        mses = [r.mse for r in results]
        maes = [r.mae for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        
        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        best_window = int(np.argmax(accuracies))
        worst_window = int(np.argmin(accuracies))
        
        total_predictions = sum(len(r.predictions) for r in results)
        
        # Generate summary
        summary = (
            f"Walk-Forward Validation Results ({len(results)} windows)\n"
            f"{'=' * 50}\n"
            f"Average Direction Accuracy: {avg_accuracy:.2%} ± {std_accuracy:.2%}\n"
            f"Average MSE: {np.mean(mses):.6f}\n"
            f"Average MAE: {np.mean(maes):.6f}\n"
            f"Average Sharpe Ratio: {np.mean(sharpes):.2f}\n"
            f"Best Window: #{best_window + 1} ({accuracies[best_window]:.2%})\n"
            f"Worst Window: #{worst_window + 1} ({accuracies[worst_window]:.2%})\n"
            f"Total Predictions: {total_predictions}"
        )
        
        return FullValidationReport(
            window_results=results,
            avg_accuracy=float(np.mean([r.accuracy for r in results])),
            avg_direction_accuracy=float(avg_accuracy),
            avg_mse=float(np.mean(mses)),
            avg_mae=float(np.mean(maes)),
            avg_sharpe=float(np.mean(sharpes)),
            std_accuracy=float(std_accuracy),
            best_window=best_window,
            worst_window=worst_window,
            total_predictions=total_predictions,
            summary=summary
        )


class BacktestSimulator:
    """
    Simulates trading based on model predictions.
    
    Provides realistic performance metrics including:
    - Cumulative returns
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001  # 0.1% per trade
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run backtest simulation.
        
        Args:
            predictions: Model predictions (0/1 for direction or probability)
            actual_returns: Actual price returns
            confidence: Optional confidence scores for position sizing
            
        Returns:
            Dictionary with backtest results
        """
        n = len(predictions)
        
        # Convert predictions to signals (-1 short, 0 flat, 1 long)
        if predictions.max() <= 1 and predictions.min() >= 0:
            # Probabilities
            signals = np.where(predictions > 0.5, 1, -1)
        else:
            signals = np.sign(predictions)
        
        # Position sizing based on confidence
        if confidence is not None:
            position_size = confidence
        else:
            position_size = np.ones(n)
        
        # Calculate returns
        strategy_returns = signals * actual_returns * position_size
        
        # Apply transaction costs (when position changes)
        position_changes = np.abs(np.diff(signals, prepend=0))
        costs = position_changes * self.transaction_cost
        strategy_returns -= costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns[-1] - 1
        
        # Sharpe ratio (annualized)
        if np.std(strategy_returns) > 0:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Win rate
        winning_trades = np.sum((strategy_returns > 0) & (signals != 0))
        total_trades = np.sum(signals != 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Compare to buy-and-hold
        buy_hold_returns = (1 + actual_returns).cumprod()
        buy_hold_total = buy_hold_returns[-1] - 1
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'buy_hold_return': float(buy_hold_total),
            'outperformance': float(total_return - buy_hold_total),
            'cumulative_returns': cumulative_returns,
            'strategy_returns': strategy_returns
        }


def compare_to_baseline(
    model_accuracy: float,
    baseline_accuracy: float = 0.5
) -> Dict:
    """
    Compare model accuracy to a baseline.
    
    Args:
        model_accuracy: Model's directional accuracy
        baseline_accuracy: Baseline accuracy (default: 50% random)
        
    Returns:
        Dictionary with comparison metrics
    """
    improvement = model_accuracy - baseline_accuracy
    relative_improvement = improvement / baseline_accuracy * 100
    
    # Statistical significance (simplified)
    # In production, use proper hypothesis testing
    significant = improvement > 0.05  # More than 5% improvement
    
    return {
        'model_accuracy': model_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'absolute_improvement': improvement,
        'relative_improvement': relative_improvement,
        'statistically_significant': significant,
        'interpretation': (
            f"Model achieves {model_accuracy:.1%} accuracy vs "
            f"{baseline_accuracy:.1%} baseline. "
            f"{'Significant' if significant else 'Not significant'} "
            f"improvement of {improvement:.1%}."
        )
    }


# Example usage
if __name__ == '__main__':
    print("Walk-Forward Validation Demo")
    print("=" * 50)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    # Simulated features and targets
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some signal (not pure random)
    signal = X[:, 0] * 0.5 + X[:, 1] * 0.3  # Linear combination of first 2 features
    noise = np.random.randn(n_samples) * 0.5
    y = (signal + noise > 0).astype(float)  # Binary classification
    
    # Create a simple model for demonstration
    class SimpleModel:
        def __init__(self):
            self.weights = None
        
        def fit(self, X, y):
            # Simple logistic regression
            # In production, use sklearn or pytorch
            self.weights = np.mean(X[y == 1], axis=0) - np.mean(X[y == 0], axis=0)
            self.weights = self.weights / (np.linalg.norm(self.weights) + 1e-8)
        
        def predict(self, X):
            scores = X @ self.weights
            return 1 / (1 + np.exp(-scores))  # Sigmoid
    
    model = SimpleModel()
    
    # Configure validator
    validator = WalkForwardValidator(
        train_window_size=200,
        test_window_size=50,
        step_size=50,
        expanding_window=False
    )
    
    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"Configuration:")
    print(f"  Train window: {validator.train_window_size}")
    print(f"  Test window: {validator.test_window_size}")
    print(f"  Step size: {validator.step_size}")
    
    # Generate splits to show
    splits = validator.get_splits(n_samples)
    print(f"\nGenerated {len(splits)} validation windows")
    
    # Run validation
    print("\nRunning validation...")
    report = validator.validate(model, X, y)
    
    print("\n" + report.summary)
    
    # Compare to baseline
    print("\n" + "-" * 50)
    comparison = compare_to_baseline(report.avg_direction_accuracy, 0.5)
    print(comparison['interpretation'])
    
    # Run backtest simulation
    print("\n" + "-" * 50)
    print("Running backtest simulation...")
    
    # Get predictions for last test window
    last_result = report.window_results[-1]
    backtester = BacktestSimulator()
    
    # Simulate returns
    actual_returns = np.random.randn(len(last_result.predictions)) * 0.02  # 2% daily std
    backtest = backtester.run_backtest(
        last_result.predictions,
        actual_returns
    )
    
    print(f"\nBacktest Results (last window):")
    print(f"  Total Return: {backtest['total_return']:.2%}")
    print(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest['max_drawdown']:.2%}")
    print(f"  Win Rate: {backtest['win_rate']:.2%}")
    print(f"  Total Trades: {backtest['total_trades']}")
    print(f"  Buy & Hold Return: {backtest['buy_hold_return']:.2%}")
    print(f"  Outperformance: {backtest['outperformance']:.2%}")
