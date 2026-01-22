"""
API Tests for Stock Predictor
=============================

Run with: pytest tests/test_api.py -v

Author: Talal Alkhaled
License: MIT
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.predict import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPI:
    """Tests for the prediction API."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Stock Predictor API"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_symbols_endpoint(self, client):
        """Test symbols listing endpoint."""
        response = client.get("/symbols")
        assert response.status_code == 200
        
        data = response.json()
        assert "symbols" in data
        assert len(data["symbols"]) > 0
        
        # Check symbol structure
        symbol = data["symbols"][0]
        assert "symbol" in symbol
        assert "name" in symbol
    
    def test_predict_endpoint_basic(self, client):
        """Test basic prediction request."""
        response = client.post(
            "/predict",
            json={"symbol": "AAPL", "timeframe": "1d"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert "current_price" in data
        assert "predictions" in data
        assert "strategies" in data
        assert "technical_indicators" in data
    
    def test_predict_endpoint_all_timeframes(self, client):
        """Test predictions include all timeframes."""
        response = client.post(
            "/predict",
            json={"symbol": "GOOGL", "timeframe": "1d"}
        )
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        expected_timeframes = ['1d', '5d', '1M', '6M', '1Y']
        for tf in expected_timeframes:
            assert tf in predictions
            assert "direction" in predictions[tf]
            assert "confidence" in predictions[tf]
            assert "price" in predictions[tf]
    
    def test_predict_endpoint_with_sentiment(self, client):
        """Test prediction with sentiment analysis."""
        response = client.post(
            "/predict",
            json={
                "symbol": "MSFT",
                "timeframe": "1d",
                "include_sentiment": True
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "sentiment" in data
        assert data["sentiment"] is not None
    
    def test_predict_endpoint_without_sentiment(self, client):
        """Test prediction without sentiment analysis."""
        response = client.post(
            "/predict",
            json={
                "symbol": "MSFT",
                "timeframe": "1d",
                "include_sentiment": False
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        # Sentiment should be None when disabled
        assert data["sentiment"] is None
    
    def test_predict_endpoint_with_patterns(self, client):
        """Test prediction with pattern detection."""
        response = client.post(
            "/predict",
            json={
                "symbol": "TSLA",
                "timeframe": "1d",
                "include_patterns": True
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "technical_patterns" in data
    
    def test_predict_invalid_timeframe(self, client):
        """Test prediction with invalid timeframe."""
        response = client.post(
            "/predict",
            json={"symbol": "AAPL", "timeframe": "invalid"}
        )
        assert response.status_code == 400
        assert "Invalid timeframe" in response.json()["detail"]
    
    def test_predict_missing_symbol(self, client):
        """Test prediction without required symbol."""
        response = client.post(
            "/predict",
            json={"timeframe": "1d"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_metrics" in data
        assert "validation_methodology" in data
        assert "baseline_comparison" in data
    
    def test_prediction_response_structure(self, client):
        """Test prediction response has correct structure."""
        response = client.post(
            "/predict",
            json={"symbol": "NVDA", "timeframe": "1d"}
        )
        assert response.status_code == 200
        
        data = response.json()
        
        # Check all required fields
        required_fields = [
            "symbol", "current_price", "previous_close",
            "change", "change_percent", "predictions",
            "strategies", "technical_indicators",
            "technical_patterns", "generated_at", "model_version"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
    
    def test_strategy_structure(self, client):
        """Test strategy objects have correct structure."""
        response = client.post(
            "/predict",
            json={"symbol": "AAPL", "timeframe": "1d"}
        )
        data = response.json()
        
        assert len(data["strategies"]) > 0
        strategy = data["strategies"][0]
        
        assert "name" in strategy
        assert "accuracy" in strategy
        assert "sharpe" in strategy
        assert "active" in strategy
        assert "reasoning" in strategy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
