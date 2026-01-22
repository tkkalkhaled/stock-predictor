"""
Stock Predictor Evaluation Package
==================================

This package contains validation and backtesting tools:

- walk_forward_validation: Time-series cross-validation

Author: Talal Alkhaled
License: MIT
"""

from .walk_forward_validation import (
    WalkForwardValidator,
    BacktestSimulator,
    ValidationResult,
    FullValidationReport,
    compare_to_baseline
)

__all__ = [
    'WalkForwardValidator',
    'BacktestSimulator',
    'ValidationResult',
    'FullValidationReport',
    'compare_to_baseline',
]
