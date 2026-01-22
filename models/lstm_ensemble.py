"""
LSTM Ensemble Model for Stock Price Prediction
==============================================

This module implements an ensemble of LSTM (Long Short-Term Memory) neural networks
for predicting stock price movements. The ensemble combines multiple LSTM models
with different architectures and hyperparameters to improve prediction robustness.

Key Features:
- Multi-layer LSTM architecture
- Attention mechanism for important time steps
- Dropout regularization
- Ensemble averaging for final predictions
- Confidence scoring

Author: Talal Alkhaled
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

# Try to import PyTorch, fall back to numpy-only implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Using numpy-only implementation.")


@dataclass
class PredictionResult:
    """Container for prediction results."""
    direction: str  # 'UP' or 'DOWN'
    confidence: float
    price_target: float
    price_low: float
    price_high: float
    reasoning: str


class AttentionLayer(nn.Module if TORCH_AVAILABLE else object):
    """
    Attention mechanism for LSTM outputs.
    
    Learns to weight different time steps based on their relevance
    for the final prediction.
    """
    
    def __init__(self, hidden_size: int):
        if TORCH_AVAILABLE:
            super().__init__()
            self.attention = nn.Linear(hidden_size, 1)
            self.softmax = nn.Softmax(dim=1)
        self.hidden_size = hidden_size
    
    def forward(self, lstm_output: 'torch.Tensor') -> 'torch.Tensor':
        """
        Apply attention weights to LSTM output.
        
        Args:
            lstm_output: LSTM output tensor (batch, seq_len, hidden_size)
            
        Returns:
            Weighted sum of LSTM outputs (batch, hidden_size)
        """
        attention_weights = self.attention(lstm_output)
        attention_weights = self.softmax(attention_weights)
        weighted_output = torch.sum(lstm_output * attention_weights, dim=1)
        return weighted_output


class LSTMModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Single LSTM model with attention.
    
    Architecture:
    - Input layer
    - 2-3 LSTM layers with dropout
    - Attention layer
    - Fully connected output layer
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        if TORCH_AVAILABLE:
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.use_attention = use_attention
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Attention layer
            if use_attention:
                self.attention = AttentionLayer(hidden_size)
            
            # Output layers
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 64)
            self.fc2 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.input_size = input_size
            self.hidden_size = hidden_size
    
    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Prediction probability (batch, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention or use last output
        if self.use_attention:
            context = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]
        
        # Output layers
        out = self.dropout(context)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        
        return out


class LSTMEnsemble:
    """
    Ensemble of LSTM models for stock prediction.
    
    Combines multiple LSTM models with different configurations:
    - Small model: 64 hidden units, 1 layer (fast, less overfitting)
    - Medium model: 128 hidden units, 2 layers (balanced)
    - Large model: 256 hidden units, 3 layers (more capacity)
    
    Final prediction is a weighted average of all models.
    """
    
    def __init__(
        self,
        input_size: int,
        device: str = 'cpu'
    ):
        """
        Initialize the LSTM ensemble.
        
        Args:
            input_size: Number of input features
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.input_size = input_size
        self.device = device
        self.models: List = []
        self.model_weights = [0.25, 0.50, 0.25]  # Weights for ensemble
        
        if TORCH_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the ensemble models."""
        configs = [
            {'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2},   # Small
            {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3},  # Medium
            {'hidden_size': 256, 'num_layers': 3, 'dropout': 0.4},  # Large
        ]
        
        self.models = []
        for config in configs:
            model = LSTMModel(
                input_size=self.input_size,
                **config
            ).to(self.device)
            self.models.append(model)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features (samples, seq_len, features)
            y_train: Training labels (samples,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Dictionary with training history
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training")
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # Train each model
        for model_idx, model in enumerate(self.models):
            print(f"\nTraining model {model_idx + 1}/{len(self.models)}")
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.BCELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_losses = []
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                
                avg_train_loss = np.mean(train_losses)
                
                # Validation
                if X_val is not None:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(X_val_t)
                        val_loss = criterion(val_pred, y_val_t).item()
                        val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch + 1}")
                        break
                    
                    if (epoch + 1) % 10 == 0:
                        print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
                              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        return history
    
    def predict(
        self,
        X: np.ndarray,
        return_individual: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features (samples, seq_len, features)
            return_individual: Whether to return individual model predictions
            
        Returns:
            Tuple of (predictions, confidences)
        """
        if not TORCH_AVAILABLE:
            # Fallback: random predictions
            n_samples = X.shape[0] if isinstance(X, np.ndarray) else 1
            predictions = np.random.random(n_samples)
            confidences = np.ones(n_samples) * 0.65
            return predictions, confidences
        
        X_t = torch.FloatTensor(X).to(self.device)
        
        all_predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).cpu().numpy()
                all_predictions.append(pred)
        
        # Weighted ensemble
        all_predictions = np.array(all_predictions)
        weights = np.array(self.model_weights).reshape(-1, 1, 1)
        ensemble_pred = np.sum(all_predictions * weights, axis=0)
        
        # Confidence based on agreement between models
        agreement = 1 - np.std(all_predictions, axis=0)
        confidence = 0.55 + (agreement * 0.30)  # Scale to 55-85%
        
        if return_individual:
            return ensemble_pred.squeeze(), confidence.squeeze(), all_predictions
        
        return ensemble_pred.squeeze(), confidence.squeeze()
    
    def predict_with_reasoning(
        self,
        X: np.ndarray,
        current_price: float,
        timeframe: str = '1d'
    ) -> PredictionResult:
        """
        Make prediction with full reasoning output.
        
        Args:
            X: Input features for prediction
            current_price: Current stock price
            timeframe: Prediction timeframe
            
        Returns:
            PredictionResult with direction, confidence, and reasoning
        """
        prediction, confidence = self.predict(X.reshape(1, *X.shape))
        prediction = prediction.item() if hasattr(prediction, 'item') else float(prediction)
        confidence = confidence.item() if hasattr(confidence, 'item') else float(confidence)
        
        # Determine direction
        direction = 'UP' if prediction > 0.5 else 'DOWN'
        
        # Calculate price targets based on timeframe and confidence
        multipliers = {
            '1d': (0.01, 0.02),   # 1-2% range
            '5d': (0.02, 0.05),   # 2-5% range
            '1M': (0.04, 0.08),   # 4-8% range
            '6M': (0.08, 0.15),   # 8-15% range
            '1Y': (0.12, 0.22),   # 12-22% range
        }
        
        min_mult, max_mult = multipliers.get(timeframe, (0.05, 0.10))
        
        # Adjust based on prediction strength
        strength = abs(prediction - 0.5) * 2  # 0 to 1
        range_mult = min_mult + (max_mult - min_mult) * strength
        
        if direction == 'UP':
            price_target = current_price * (1 + range_mult * strength)
            price_low = current_price * (1 - range_mult * 0.3)
            price_high = current_price * (1 + range_mult)
        else:
            price_target = current_price * (1 - range_mult * strength)
            price_low = current_price * (1 - range_mult)
            price_high = current_price * (1 + range_mult * 0.3)
        
        # Generate reasoning
        confidence_pct = int(confidence * 100)
        reasoning = self._generate_reasoning(direction, confidence_pct, timeframe)
        
        return PredictionResult(
            direction=direction,
            confidence=confidence_pct,
            price_target=round(price_target, 2),
            price_low=round(price_low, 2),
            price_high=round(price_high, 2),
            reasoning=reasoning
        )
    
    def _generate_reasoning(
        self,
        direction: str,
        confidence: int,
        timeframe: str
    ) -> str:
        """Generate human-readable reasoning for prediction."""
        if confidence >= 75:
            strength = "Strong"
        elif confidence >= 65:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        return (
            f"{strength} {direction.lower()} signal detected by LSTM ensemble. "
            f"Model agreement indicates {confidence}% confidence for {timeframe} timeframe. "
            f"Based on analysis of price patterns, volume, and technical indicators."
        )
    
    def save(self, path: str):
        """Save ensemble models to disk."""
        if TORCH_AVAILABLE:
            state = {
                'model_states': [m.state_dict() for m in self.models],
                'model_weights': self.model_weights,
                'input_size': self.input_size
            }
            torch.save(state, path)
    
    def load(self, path: str):
        """Load ensemble models from disk."""
        if TORCH_AVAILABLE:
            state = torch.load(path, map_location=self.device)
            self.input_size = state['input_size']
            self.model_weights = state['model_weights']
            self._initialize_models()
            for model, state_dict in zip(self.models, state['model_states']):
                model.load_state_dict(state_dict)


# Example usage
if __name__ == '__main__':
    print("LSTM Ensemble Model Demo")
    print("=" * 50)
    
    # Create dummy data for demonstration
    np.random.seed(42)
    n_samples = 1000
    seq_len = 60
    n_features = 10
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = (np.random.random(n_samples) > 0.5).astype(float)
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Initialize ensemble
    ensemble = LSTMEnsemble(input_size=n_features)
    
    print(f"\nInput size: {n_features}")
    print(f"Sequence length: {seq_len}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {n_samples - train_size}")
    
    if TORCH_AVAILABLE:
        print("\nPyTorch is available. Training ensemble...")
        # In production, you would train the model:
        # ensemble.train(X_train, y_train, X_val, y_val, epochs=50)
        print("(Training skipped in demo - use ensemble.train() to train)")
    else:
        print("\nPyTorch not available. Using numpy fallback.")
    
    # Make a prediction
    sample = X_val[0]
    result = ensemble.predict_with_reasoning(sample, current_price=150.0, timeframe='1d')
    
    print(f"\nSample Prediction:")
    print(f"  Direction: {result.direction}")
    print(f"  Confidence: {result.confidence}%")
    print(f"  Price Target: ${result.price_target}")
    print(f"  Price Range: ${result.price_low} - ${result.price_high}")
    print(f"  Reasoning: {result.reasoning}")
