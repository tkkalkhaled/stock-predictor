"""
Stock Predictor Data Package
============================

This package handles data ingestion and preprocessing:

- preprocess: Data cleaning, normalization, and feature engineering

Author: Talal Alkhaled
License: MIT
"""

from .preprocess import DataPreprocessor, StockData

__all__ = [
    'DataPreprocessor',
    'StockData',
]
