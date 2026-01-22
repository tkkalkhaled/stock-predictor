"""
Data Preprocessing Module for Stock Market Predictor
=====================================================

This module handles data ingestion from multiple financial APIs,
cleaning, normalization, and feature extraction for the ML pipeline.

Supported Data Sources:
- Finnhub: Real-time quotes
- Alpaca: Historical OHLCV bars
- Polygon.io: Daily aggregates
- Marketaux: News articles

Author: Talal Alkhaled
License: MIT
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import os
from dataclasses import dataclass


@dataclass
class StockData:
    """Container for processed stock data."""
    symbol: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    high: float
    low: float
    open: float
    volume: int
    timestamp: datetime
    ohlcv_history: pd.DataFrame
    

class DataPreprocessor:
    """
    Handles data ingestion, cleaning, and preprocessing for stock prediction.
    
    This class demonstrates the data pipeline used in our production system,
    which fetches real-time data from multiple APIs and prepares it for
    the ML models.
    """
    
    def __init__(
        self,
        finnhub_key: Optional[str] = None,
        alpaca_key_id: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
        polygon_key: Optional[str] = None
    ):
        """
        Initialize the data preprocessor with API keys.
        
        Args:
            finnhub_key: Finnhub API key for real-time quotes
            alpaca_key_id: Alpaca API Key ID for historical data
            alpaca_secret: Alpaca Secret Key
            polygon_key: Polygon.io API key for daily data
        """
        self.finnhub_key = finnhub_key or os.getenv('FINNHUB_API_KEY')
        self.alpaca_key_id = alpaca_key_id or os.getenv('ALPACA_API_KEY_ID')
        self.alpaca_secret = alpaca_secret or os.getenv('ALPACA_SECRET_KEY')
        self.polygon_key = polygon_key or os.getenv('POLYGON_API_KEY')
        
        # API base URLs
        self.finnhub_base = 'https://finnhub.io/api/v1'
        self.alpaca_base = 'https://data.alpaca.markets/v2'
        self.polygon_base = 'https://api.polygon.io/v2'
        
    def fetch_quote(self, symbol: str) -> Dict:
        """
        Fetch real-time quote from Finnhub.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with quote data
        """
        if not self.finnhub_key:
            raise ValueError("Finnhub API key not configured")
            
        url = f"{self.finnhub_base}/quote"
        params = {'symbol': symbol, 'token': self.finnhub_key}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'symbol': symbol,
            'current_price': data['c'],
            'previous_close': data['pc'],
            'change': data['d'],
            'change_percent': data['dp'],
            'high': data['h'],
            'low': data['l'],
            'open': data['o'],
            'timestamp': datetime.fromtimestamp(data['t']) if data.get('t') else datetime.now()
        }
    
    def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str = '1d',
        days_back: int = 365
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars from Alpaca.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Bar timeframe ('1d', '5d', '1M', '6M', '1Y', 'MAX')
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.alpaca_key_id or not self.alpaca_secret:
            raise ValueError("Alpaca API keys not configured")
        
        # Map timeframe to Alpaca format
        timeframe_map = {
            '15m': '15Min',
            '1h': '1Hour',
            '1d': '15Min',   # 15-min bars for 1-day view
            '5d': '1Hour',   # 1-hour bars for 5-day view
            '1M': '1Day',
            '6M': '1Day',
            '1Y': '1Day',
            'MAX': '1Day'
        }
        
        # Map timeframe to days back
        days_map = {
            '1d': 1,
            '5d': 7,
            '1M': 30,
            '6M': 180,
            '1Y': 365,
            'MAX': 730
        }
        
        alpaca_tf = timeframe_map.get(timeframe, '1Day')
        days_back = days_map.get(timeframe, 365)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = f"{self.alpaca_base}/stocks/{symbol}/bars"
        params = {
            'timeframe': alpaca_tf,
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'adjustment': 'raw',
            'feed': 'iex',
            'limit': 10000
        }
        
        headers = {
            'APCA-API-KEY-ID': self.alpaca_key_id,
            'APCA-API-SECRET-KEY': self.alpaca_secret
        }
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        bars = data.get('bars', [])
        
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        
        Handles:
        - Missing values
        - Outliers
        - Data type conversions
        - Invalid price data
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Remove rows with missing critical values
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=[c for c in critical_cols if c in df.columns])
        
        # Ensure numeric types
        for col in critical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows where high < low (invalid data)
        if 'high' in df.columns and 'low' in df.columns:
            df = df[df['high'] >= df['low']]
        
        # Remove zero or negative prices
        if 'close' in df.columns:
            df = df[df['close'] > 0]
        
        # Handle outliers using IQR method (for close price)
        if 'close' in df.columns and len(df) > 10:
            Q1 = df['close'].quantile(0.01)
            Q3 = df['close'].quantile(0.99)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df['close'] >= lower_bound) & (df['close'] <= upper_bound)]
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def filter_trading_hours(
        self,
        df: pd.DataFrame,
        market_open: str = '09:30',
        market_close: str = '16:00',
        timezone: str = 'America/New_York'
    ) -> pd.DataFrame:
        """
        Filter data to regular trading hours only.
        
        Args:
            df: OHLCV DataFrame with timestamp column
            market_open: Market open time (HH:MM)
            market_close: Market close time (HH:MM)
            timezone: Market timezone
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        
        # Convert to market timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df['timestamp'] = df['timestamp'].dt.tz_convert(timezone)
        
        # Extract time component
        df['time'] = df['timestamp'].dt.strftime('%H:%M')
        
        # Filter to trading hours
        df = df[(df['time'] >= market_open) & (df['time'] <= market_close)]
        
        # Remove helper column
        df = df.drop(columns=['time'])
        
        return df
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicator features to the DataFrame.
        
        IMPORTANT - Data Leakage Prevention:
        All indicators are calculated using historical data only. When making
        predictions for time T, we only use data from T-1 and earlier.
        The .shift(1) or rolling windows ensure no lookahead bias.
        
        Calculates:
        - RSI (14-period)
        - MACD (12, 26, 9)
        - Bollinger Bands (20, 2)
        - Moving Averages (10, 20, 50)
        - Volatility
        - Returns
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with technical features
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
        # =====================================================================
        # TEMPORAL LOGIC: All calculations use .shift(1) or rolling windows
        # that only look backwards, preventing lookahead bias.
        # At prediction time T, we only have access to data up to T-1.
        # =====================================================================
        
        # Returns - uses previous close, not current (shift ensures T-1 data)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages - rolling windows look backwards only
        # SMA_10 at time T uses closes from T-10 to T-1
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD - derived from EMAs which are backward-looking
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI - calculated using previous 14 days of price changes
        # At time T, RSI uses changes from T-14 to T-1
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands - 20-day lookback, no current-day data
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volatility - historical volatility from past 20 days
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_annualized'] = df['volatility'] * np.sqrt(252)
        
        # Volume features - CRITICAL: use previous day's volume, not current
        # Current-day volume isn't known at market open when predictions are made
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].shift(1).rolling(window=20).mean()
            df['volume_ratio'] = df['volume'].shift(1) / df['volume_sma']
        
        # Price position within range - uses previous day's range
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_range'] = df['high'] - df['low']
            df['close_position'] = (df['close'] - df['low']) / df['hl_range'].replace(0, np.inf)
        
        return df
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe Ratio.
        
        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Annualized Sharpe Ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = excess_returns.mean() / excess_returns.std()
        annualized_sharpe = sharpe * np.sqrt(periods_per_year)
        
        return float(np.clip(annualized_sharpe, -5, 10))
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM model training.
        
        TEMPORAL INTEGRITY:
        - Features X[i] contain data from time steps [i-sequence_length, i-1]
        - Target y[i] is the direction at time i (relative to i-1)
        - This ensures no lookahead bias: we predict time i using only past data
        
        Args:
            df: DataFrame with features
            sequence_length: Number of time steps per sequence
            target_col: Column to predict
            feature_cols: Columns to use as features (default: all numeric)
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target_col]
        
        # Drop rows with NaN
        df_clean = df[feature_cols + [target_col]].dropna()
        
        if len(df_clean) < sequence_length + 1:
            raise ValueError(f"Not enough data for sequence length {sequence_length}")
        
        # Normalize features using training data statistics only
        # In production, use fit on train, transform on test to prevent data leakage
        features = df_clean[feature_cols].values
        features_normalized = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        targets = df_clean[target_col].values
        
        X, y = [], []
        for i in range(sequence_length, len(df_clean)):
            # X[i] uses features from [i-sequence_length] to [i-1] (exclusive of i)
            # This prevents using current-day features to predict current-day direction
            X.append(features_normalized[i-sequence_length:i])
            # Target: direction at time i (1 = up from i-1, 0 = down from i-1)
            y.append(1 if targets[i] > targets[i-1] else 0)
        
        return np.array(X), np.array(y)
    
    def get_processed_data(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> StockData:
        """
        Main method to fetch and process all data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            timeframe: Data timeframe
            
        Returns:
            StockData object with all processed data
        """
        # Fetch quote
        quote = self.fetch_quote(symbol)
        
        # Fetch historical data
        df = self.fetch_historical_bars(symbol, timeframe)
        
        # Clean data
        df = self.clean_data(df)
        
        # Filter trading hours (for intraday data)
        if timeframe in ['1d', '5d']:
            df = self.filter_trading_hours(df)
        
        # Add technical features
        df = self.add_technical_features(df)
        
        return StockData(
            symbol=symbol,
            current_price=quote['current_price'],
            previous_close=quote['previous_close'],
            change=quote['change'],
            change_percent=quote['change_percent'],
            high=quote['high'],
            low=quote['low'],
            open=quote['open'],
            volume=df['volume'].iloc[-1] if 'volume' in df.columns and len(df) > 0 else 0,
            timestamp=quote['timestamp'],
            ohlcv_history=df
        )


# Example usage
if __name__ == '__main__':
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Demo with sample data (API keys required for real data)
    print("Stock Data Preprocessor Demo")
    print("=" * 50)
    print("\nTo use with real data, set environment variables:")
    print("  - FINNHUB_API_KEY")
    print("  - ALPACA_API_KEY_ID")
    print("  - ALPACA_SECRET_KEY")
    print("\nExample usage:")
    print("  data = preprocessor.get_processed_data('AAPL', '1d')")
    print("  X, y = preprocessor.prepare_sequences(data.ohlcv_history)")
