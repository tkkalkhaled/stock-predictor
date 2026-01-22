"""
Configuration Management for Stock Predictor
=============================================

Centralizes all configuration settings and environment variable loading.

Author: Talal Alkhaled
License: MIT
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class APIConfig:
    """Configuration for external APIs."""
    finnhub_key: Optional[str] = field(default_factory=lambda: os.getenv('FINNHUB_API_KEY'))
    alpaca_key_id: Optional[str] = field(default_factory=lambda: os.getenv('ALPACA_API_KEY_ID'))
    alpaca_secret: Optional[str] = field(default_factory=lambda: os.getenv('ALPACA_SECRET_KEY'))
    polygon_key: Optional[str] = field(default_factory=lambda: os.getenv('POLYGON_API_KEY'))
    marketaux_key: Optional[str] = field(default_factory=lambda: os.getenv('MARKETAUX_API_KEY'))
    openai_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # LSTM settings
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Validation settings
    train_window_size: int = 252  # ~1 year of trading days
    test_window_size: int = 21    # ~1 month
    step_size: int = 21
    
    # Model paths
    model_dir: Path = field(default_factory=lambda: Path('models/saved'))
    lstm_model_path: Path = field(default_factory=lambda: Path('models/saved/lstm_ensemble.pt'))


@dataclass
class ServerConfig:
    """Configuration for API server."""
    host: str = field(default_factory=lambda: os.getenv('API_HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.getenv('API_PORT', '8000')))
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    cors_origins: list = field(default_factory=lambda: ['*'])


@dataclass
class Config:
    """Main configuration class."""
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / 'data')
    
    def __post_init__(self):
        """Ensure required directories exist."""
        self.model.model_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_api_keys(self) -> dict:
        """Check which API keys are configured."""
        return {
            'finnhub': bool(self.api.finnhub_key),
            'alpaca': bool(self.api.alpaca_key_id and self.api.alpaca_secret),
            'polygon': bool(self.api.polygon_key),
            'marketaux': bool(self.api.marketaux_key),
            'openai': bool(self.api.openai_key),
        }
    
    def print_status(self):
        """Print configuration status."""
        print("Stock Predictor Configuration")
        print("=" * 40)
        print(f"\nProject root: {self.project_root}")
        print(f"Model directory: {self.model.model_dir}")
        
        print("\nAPI Keys Status:")
        for api, configured in self.validate_api_keys().items():
            status = "[OK] Configured" if configured else "[--] Not set"
            print(f"  {api}: {status}")
        
        print(f"\nServer: {self.server.host}:{self.server.port}")
        print(f"Debug mode: {self.server.debug}")


# Global configuration instance
config = Config()


# Supported stock symbols (can be extended)
SUPPORTED_SYMBOLS = [
    'AAPL', 'GOOGL', 'GOOG', 'MSFT', 'TSLA', 'NVDA', 
    'AMZN', 'META', 'NFLX', 'AMD', 'INTC', 'CRM',
    'JPM', 'BAC', 'GS', 'V', 'MA',
    'JNJ', 'PFE', 'UNH',
    'XOM', 'CVX',
    'DIS', 'NKE', 'SBUX',
    'SPY', 'QQQ', 'IWM'  # ETFs
]


# Timeframe configurations
TIMEFRAMES = {
    '1d': {
        'alpaca_timeframe': '15Min',
        'days_back': 1,
        'description': '1 Day',
        'expected_accuracy': 68,
    },
    '5d': {
        'alpaca_timeframe': '1Hour',
        'days_back': 7,
        'description': '5 Days',
        'expected_accuracy': 72,
    },
    '1M': {
        'alpaca_timeframe': '1Day',
        'days_back': 30,
        'description': '1 Month',
        'expected_accuracy': 65,
    },
    '6M': {
        'alpaca_timeframe': '1Day',
        'days_back': 180,
        'description': '6 Months',
        'expected_accuracy': 70,
    },
    '1Y': {
        'alpaca_timeframe': '1Day',
        'days_back': 365,
        'description': '1 Year',
        'expected_accuracy': 75,
    },
}


if __name__ == '__main__':
    config.print_status()
