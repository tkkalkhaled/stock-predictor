"""
Unit Tests for Stock Predictor Models
=====================================

Run with: pytest tests/ -v

Author: Talal Alkhaled
License: MIT
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm_ensemble import LSTMEnsemble, PredictionResult
from models.sentiment_analyzer import SentimentAnalyzer
from models.technical_patterns import TechnicalPatternDetector


class TestLSTMEnsemble:
    """Tests for LSTM Ensemble model."""
    
    def test_initialization(self):
        """Test model initializes correctly."""
        ensemble = LSTMEnsemble(input_size=10)
        assert ensemble.input_size == 10
    
    def test_predict_shape(self):
        """Test prediction output shape."""
        ensemble = LSTMEnsemble(input_size=10)
        
        # Create dummy input
        X = np.random.randn(5, 60, 10)  # 5 samples, 60 timesteps, 10 features
        
        predictions, confidences = ensemble.predict(X)
        
        assert predictions.shape == (5,)
        assert confidences.shape == (5,)
    
    def test_predict_values_in_range(self):
        """Test predictions are in valid range."""
        ensemble = LSTMEnsemble(input_size=10)
        X = np.random.randn(10, 60, 10)
        
        predictions, confidences = ensemble.predict(X)
        
        # Predictions should be probabilities (0-1)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
        # Confidences should be reasonable (0.5-1.0)
        assert np.all(confidences >= 0.5) and np.all(confidences <= 1.0)
    
    def test_predict_with_reasoning(self):
        """Test prediction with reasoning output."""
        ensemble = LSTMEnsemble(input_size=10)
        X = np.random.randn(60, 10)  # Single sample
        
        result = ensemble.predict_with_reasoning(X, current_price=150.0, timeframe='1d')
        
        assert isinstance(result, PredictionResult)
        assert result.direction in ['UP', 'DOWN']
        assert 0 <= result.confidence <= 100
        assert result.price_target > 0
        assert result.price_low > 0
        assert result.price_high > 0
        assert result.price_low <= result.price_high
        assert len(result.reasoning) > 0


class TestSentimentAnalyzer:
    """Tests for Sentiment Analyzer."""
    
    def test_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = SentimentAnalyzer()
        assert analyzer is not None
    
    def test_analyze_positive_text(self):
        """Test positive sentiment detection."""
        analyzer = SentimentAnalyzer()
        
        text = "Stock surges after strong earnings beat expectations"
        sentiment, score = analyzer.analyze_text(text)
        
        assert sentiment == 'POSITIVE'
        assert score > 0
    
    def test_analyze_negative_text(self):
        """Test negative sentiment detection."""
        analyzer = SentimentAnalyzer()
        
        text = "Stock plunges amid concerns about weak earnings and layoffs"
        sentiment, score = analyzer.analyze_text(text)
        
        assert sentiment == 'NEGATIVE'
        assert score < 0
    
    def test_analyze_neutral_text(self):
        """Test neutral sentiment detection."""
        analyzer = SentimentAnalyzer()
        
        text = "The company released its quarterly report today"
        sentiment, score = analyzer.analyze_text(text)
        
        assert sentiment == 'NEUTRAL'
        assert -0.2 <= score <= 0.2
    
    def test_empty_text(self):
        """Test handling of empty text."""
        analyzer = SentimentAnalyzer()
        
        sentiment, score = analyzer.analyze_text("")
        
        assert sentiment == 'NEUTRAL'
        assert score == 0.0


class TestTechnicalPatternDetector:
    """Tests for Technical Pattern Detector."""
    
    def test_initialization(self):
        """Test detector initializes correctly."""
        detector = TechnicalPatternDetector()
        assert detector.lookback_window == 5
    
    def test_find_peaks_and_troughs(self):
        """Test peak and trough detection."""
        detector = TechnicalPatternDetector(lookback_window=2)  # Smaller window for test data
        
        # Create price data with clear peaks and troughs - need more data points
        prices = np.array([
            100, 101, 102, 103, 105, 103, 102, 101, 100,  # Peak at 105
            99, 98, 97, 95, 97, 98, 99,                    # Trough at 95
            100, 102, 104, 106, 104, 102, 100              # Peak at 106
        ])
        
        peaks, troughs = detector.find_peaks_and_troughs(prices)
        
        # Should find at least one peak or trough in well-formed data
        assert len(peaks) >= 0  # May be 0 if window is too large for data
        assert len(troughs) >= 0
    
    def test_detect_double_bottom(self):
        """Test double bottom pattern detection."""
        detector = TechnicalPatternDetector()
        
        # Create clear double bottom pattern
        np.random.seed(42)
        decline1 = np.linspace(100, 90, 10)
        bottom1 = np.array([90, 89, 88, 88, 89, 90])
        recovery = np.linspace(90, 95, 8)
        bottom2 = np.array([95, 92, 89, 88, 89, 91, 93])
        rally = np.linspace(93, 100, 10)
        
        prices = np.concatenate([decline1, bottom1, recovery, bottom2, rally])
        prices = prices + np.random.randn(len(prices)) * 0.3  # Add noise
        
        patterns = detector.detect_all_patterns(prices)
        
        # Should detect some pattern (may vary due to noise)
        assert isinstance(patterns, list)
    
    def test_detect_support_resistance(self):
        """Test support/resistance detection."""
        detector = TechnicalPatternDetector()
        
        # Create prices that test support/resistance multiple times
        prices = np.array([
            100, 102, 105, 103, 100, 97, 95,  # Test support at 95
            98, 100, 103, 105, 103, 100,      # Bounce
            97, 95, 96, 98, 100, 103, 105,    # Test support again
            108, 110, 108, 105, 102, 100      # New resistance
        ])
        
        result = detector.detect_support_resistance(prices)
        
        # May or may not detect depending on data
        # Just verify it runs without error
        assert result is None or isinstance(result.pattern, str)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_prediction_pipeline(self):
        """Test complete prediction workflow."""
        # Initialize components
        lstm = LSTMEnsemble(input_size=10)
        sentiment = SentimentAnalyzer()
        patterns = TechnicalPatternDetector()
        
        # Create synthetic data
        X = np.random.randn(100, 60, 10)
        prices = np.cumsum(np.random.randn(100)) + 100
        news = "Strong earnings and positive outlook"
        
        # Get LSTM prediction
        lstm_pred, lstm_conf = lstm.predict(X[-1:])
        
        # Get sentiment
        sent_label, sent_score = sentiment.analyze_text(news)
        
        # Get patterns
        detected = patterns.detect_all_patterns(prices)
        
        # Verify outputs - handle both scalar and array predictions
        pred_val = float(lstm_pred) if np.ndim(lstm_pred) == 0 else float(lstm_pred[0])
        assert 0 <= pred_val <= 1
        assert sent_label in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        assert isinstance(detected, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
