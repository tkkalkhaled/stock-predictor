"""
Technical Pattern Recognition Module
====================================

This module detects chart patterns in stock price data that can
indicate potential price movements. Pattern recognition is a key
component of technical analysis.

Supported Patterns:
- Head and Shoulders (Bearish reversal)
- Inverse Head and Shoulders (Bullish reversal)
- Double Top (Bearish reversal)
- Double Bottom (Bullish reversal)
- Ascending Triangle (Bullish continuation)
- Descending Triangle (Bearish continuation)
- Symmetrical Triangle (Continuation, direction varies)
- Support and Resistance Levels

Author: Talal Alkhaled
License: MIT
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class PatternResult:
    """Container for detected pattern."""
    pattern: str
    signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: int  # 0-100
    description: str
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None


class TechnicalPatternDetector:
    """
    Detects technical chart patterns from price data.
    
    Uses a combination of:
    - Peak and trough detection
    - Trend line analysis
    - Statistical validation
    
    Each pattern is assigned a confidence score based on
    how well the price action matches the ideal pattern.
    """
    
    def __init__(self, lookback_window: int = 5):
        """
        Initialize the pattern detector.
        
        Args:
            lookback_window: Window size for peak/trough detection
        """
        self.lookback_window = lookback_window
    
    def detect_all_patterns(
        self,
        prices: np.ndarray,
        return_top_n: int = 3
    ) -> List[PatternResult]:
        """
        Detect all patterns in price data.
        
        Args:
            prices: Array of closing prices
            return_top_n: Number of top patterns to return
            
        Returns:
            List of detected patterns sorted by confidence
        """
        if len(prices) < 15:
            return []
        
        patterns = []
        
        # Try each pattern detector
        pattern_detectors = [
            self.detect_head_and_shoulders,
            self.detect_inverse_head_and_shoulders,
            self.detect_double_top,
            self.detect_double_bottom,
            self.detect_triangles,
            self.detect_support_resistance
        ]
        
        for detector in pattern_detectors:
            result = detector(prices)
            if result:
                patterns.append(result)
        
        # Sort by confidence and return top N
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        return patterns[:return_top_n]
    
    def find_peaks_and_troughs(
        self,
        prices: np.ndarray
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Find local peaks and troughs in price data.
        
        Args:
            prices: Array of prices
            
        Returns:
            Tuple of (peaks, troughs) where each is a list of
            dictionaries with 'index' and 'price' keys
        """
        peaks = []
        troughs = []
        lookback = self.lookback_window
        
        for i in range(lookback, len(prices) - lookback):
            window = prices[i - lookback:i + lookback + 1]
            current = prices[i]
            
            # Check if current is the highest in window (peak)
            if current == np.max(window):
                peaks.append({'index': i, 'price': float(current)})
            
            # Check if current is the lowest in window (trough)
            if current == np.min(window):
                troughs.append({'index': i, 'price': float(current)})
        
        return peaks, troughs
    
    def detect_head_and_shoulders(
        self,
        prices: np.ndarray
    ) -> Optional[PatternResult]:
        """
        Detect Head and Shoulders pattern.
        
        Pattern: Three peaks where middle peak (head) is highest,
        and two shoulders are roughly equal in height.
        
        Signal: Bearish reversal
        """
        peaks, _ = self.find_peaks_and_troughs(prices)
        
        if len(peaks) < 3:
            return None
        
        # Look for three consecutive peaks
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Head must be highest
            if not (head['price'] > left_shoulder['price'] and 
                    head['price'] > right_shoulder['price']):
                continue
            
            # Shoulders should be similar (within 5% of each other)
            shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
            avg_shoulder = (left_shoulder['price'] + right_shoulder['price']) / 2
            
            if shoulder_diff / avg_shoulder < 0.05:
                # Calculate confidence based on pattern quality
                head_prominence = (head['price'] - avg_shoulder) / avg_shoulder
                confidence = min(75, 55 + int(head_prominence * 100))
                
                return PatternResult(
                    pattern='Head & Shoulders',
                    signal='BEARISH',
                    confidence=confidence,
                    description='Reversal pattern - potential downtrend ahead',
                    resistance_level=head['price']
                )
        
        return None
    
    def detect_inverse_head_and_shoulders(
        self,
        prices: np.ndarray
    ) -> Optional[PatternResult]:
        """
        Detect Inverse Head and Shoulders pattern.
        
        Pattern: Three troughs where middle trough (head) is lowest,
        and two shoulders are roughly equal.
        
        Signal: Bullish reversal
        """
        _, troughs = self.find_peaks_and_troughs(prices)
        
        if len(troughs) < 3:
            return None
        
        # Look for three consecutive troughs
        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]
            head = troughs[i + 1]
            right_shoulder = troughs[i + 2]
            
            # Head must be lowest
            if not (head['price'] < left_shoulder['price'] and 
                    head['price'] < right_shoulder['price']):
                continue
            
            # Shoulders should be similar
            shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
            avg_shoulder = (left_shoulder['price'] + right_shoulder['price']) / 2
            
            if shoulder_diff / avg_shoulder < 0.05:
                head_depth = (avg_shoulder - head['price']) / avg_shoulder
                confidence = min(75, 55 + int(head_depth * 100))
                
                return PatternResult(
                    pattern='Inverse Head & Shoulders',
                    signal='BULLISH',
                    confidence=confidence,
                    description='Reversal pattern - potential uptrend ahead',
                    support_level=head['price']
                )
        
        return None
    
    def detect_double_top(
        self,
        prices: np.ndarray
    ) -> Optional[PatternResult]:
        """
        Detect Double Top pattern.
        
        Pattern: Two peaks at similar price levels,
        indicating strong resistance.
        
        Signal: Bearish reversal
        """
        peaks, _ = self.find_peaks_and_troughs(prices)
        
        if len(peaks) < 2:
            return None
        
        # Look for two peaks at similar levels
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            
            # Peaks should be similar (within 3%)
            price_diff = abs(peak1['price'] - peak2['price'])
            avg_price = (peak1['price'] + peak2['price']) / 2
            
            # Peaks should be separated by at least 5 periods
            index_diff = peak2['index'] - peak1['index']
            
            if price_diff / avg_price < 0.03 and index_diff >= 5:
                confidence = 58 - int(price_diff / avg_price * 100)
                
                return PatternResult(
                    pattern='Double Top',
                    signal='BEARISH',
                    confidence=confidence,
                    description='Resistance level - potential reversal down',
                    resistance_level=avg_price
                )
        
        return None
    
    def detect_double_bottom(
        self,
        prices: np.ndarray
    ) -> Optional[PatternResult]:
        """
        Detect Double Bottom pattern.
        
        Pattern: Two troughs at similar price levels,
        indicating strong support.
        
        Signal: Bullish reversal
        """
        _, troughs = self.find_peaks_and_troughs(prices)
        
        if len(troughs) < 2:
            return None
        
        for i in range(len(troughs) - 1):
            trough1 = troughs[i]
            trough2 = troughs[i + 1]
            
            price_diff = abs(trough1['price'] - trough2['price'])
            avg_price = (trough1['price'] + trough2['price']) / 2
            index_diff = trough2['index'] - trough1['index']
            
            if price_diff / avg_price < 0.03 and index_diff >= 5:
                confidence = 58 - int(price_diff / avg_price * 100)
                
                return PatternResult(
                    pattern='Double Bottom',
                    signal='BULLISH',
                    confidence=confidence,
                    description='Support level - potential reversal up',
                    support_level=avg_price
                )
        
        return None
    
    def detect_triangles(
        self,
        prices: np.ndarray
    ) -> Optional[PatternResult]:
        """
        Detect Triangle patterns (Ascending, Descending, Symmetrical).
        
        - Ascending: Rising troughs, flat peaks (Bullish)
        - Descending: Falling peaks, flat troughs (Bearish)
        - Symmetrical: Converging peaks and troughs (Neutral)
        """
        peaks, troughs = self.find_peaks_and_troughs(prices)
        
        if len(peaks) < 2 or len(troughs) < 2:
            return None
        
        # Get recent peaks and troughs
        recent_peaks = peaks[-4:] if len(peaks) >= 4 else peaks
        recent_troughs = troughs[-4:] if len(troughs) >= 4 else troughs
        
        if len(recent_peaks) < 2 or len(recent_troughs) < 2:
            return None
        
        # Calculate trend lines
        peak_trend = self._calculate_trend(recent_peaks)
        trough_trend = self._calculate_trend(recent_troughs)
        
        peak_std = np.std([p['price'] for p in recent_peaks])
        trough_std = np.std([t['price'] for t in recent_troughs])
        
        # Ascending triangle: rising troughs, flat peaks
        if trough_trend > 0.001 and peak_std < abs(peak_trend) * 0.5:
            return PatternResult(
                pattern='Ascending Triangle',
                signal='BULLISH',
                confidence=60,
                description='Bullish continuation pattern'
            )
        
        # Descending triangle: falling peaks, flat troughs
        if peak_trend < -0.001 and trough_std < abs(trough_trend) * 0.5:
            return PatternResult(
                pattern='Descending Triangle',
                signal='BEARISH',
                confidence=60,
                description='Bearish continuation pattern'
            )
        
        # Symmetrical triangle: converging
        if peak_trend < 0 and trough_trend > 0:
            return PatternResult(
                pattern='Symmetrical Triangle',
                signal='NEUTRAL',
                confidence=55,
                description='Consolidation - wait for breakout'
            )
        
        return None
    
    def _calculate_trend(self, points: List[Dict]) -> float:
        """Calculate trend slope from a list of points."""
        if len(points) < 2:
            return 0.0
        
        indices = [p['index'] for p in points]
        prices = [p['price'] for p in points]
        
        # Simple linear regression slope
        n = len(points)
        sum_x = sum(indices)
        sum_y = sum(prices)
        sum_xy = sum(x * y for x, y in zip(indices, prices))
        sum_xx = sum(x * x for x in indices)
        
        denominator = n * sum_xx - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def detect_support_resistance(
        self,
        prices: np.ndarray
    ) -> Optional[PatternResult]:
        """
        Detect key support and resistance levels.
        
        Uses clustering of peaks and troughs to identify
        significant price levels.
        """
        peaks, troughs = self.find_peaks_and_troughs(prices)
        
        if len(peaks) + len(troughs) < 3:
            return None
        
        # Combine and cluster price levels
        all_levels = [p['price'] for p in peaks] + [t['price'] for t in troughs]
        all_levels.sort()
        
        # Find clusters (prices within 2% of each other)
        clusters = []
        current_cluster = [all_levels[0]]
        
        for price in all_levels[1:]:
            if price <= current_cluster[-1] * 1.02:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [price]
        
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        if not clusters:
            return None
        
        # Find strongest support (below current price) and resistance (above)
        current_price = prices[-1]
        support = None
        resistance = None
        
        for cluster in clusters:
            avg_level = np.mean(cluster)
            strength = len(cluster)
            
            if avg_level < current_price:
                if support is None or strength > support[1]:
                    support = (avg_level, strength)
            else:
                if resistance is None or strength > resistance[1]:
                    resistance = (avg_level, strength)
        
        if support or resistance:
            signal = 'BULLISH' if support else 'BEARISH'
            desc_parts = []
            support_val = None
            resistance_val = None
            
            if support:
                support_val = support[0]
                desc_parts.append(f"Support at ${support[0]:.2f}")
            if resistance:
                resistance_val = resistance[0]
                desc_parts.append(f"Resistance at ${resistance[0]:.2f}")
            
            return PatternResult(
                pattern='Support/Resistance',
                signal=signal,
                confidence=55,
                description=' | '.join(desc_parts),
                support_level=support_val,
                resistance_level=resistance_val
            )
        
        return None
    
    def get_pattern_signals(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Get pattern-based trading signals from a DataFrame.
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Dictionary with signals and pattern details
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        prices = df['close'].values
        patterns = self.detect_all_patterns(prices)
        
        # Aggregate signal
        if not patterns:
            return {
                'signal': 'NEUTRAL',
                'confidence': 50,
                'patterns': [],
                'reasoning': 'No significant patterns detected'
            }
        
        # Weight patterns by confidence
        bullish_score = sum(p.confidence for p in patterns if p.signal == 'BULLISH')
        bearish_score = sum(p.confidence for p in patterns if p.signal == 'BEARISH')
        
        if bullish_score > bearish_score + 10:
            signal = 'BULLISH'
            confidence = min(75, int(bullish_score / len(patterns)))
        elif bearish_score > bullish_score + 10:
            signal = 'BEARISH'
            confidence = min(75, int(bearish_score / len(patterns)))
        else:
            signal = 'NEUTRAL'
            confidence = 50
        
        pattern_names = [p.pattern for p in patterns]
        
        return {
            'signal': signal,
            'confidence': confidence,
            'patterns': [
                {
                    'pattern': p.pattern,
                    'signal': p.signal,
                    'confidence': p.confidence,
                    'description': p.description
                }
                for p in patterns
            ],
            'reasoning': f"Detected patterns: {', '.join(pattern_names)}"
        }


# Example usage
if __name__ == '__main__':
    print("Technical Pattern Detector Demo")
    print("=" * 50)
    
    # Generate sample price data with patterns
    np.random.seed(42)
    n = 100
    
    # Create price data with a double bottom pattern
    base_trend = np.linspace(100, 95, 30)  # Initial decline
    bottom1 = np.array([95, 94, 93, 92, 93, 94])  # First bottom
    middle = np.linspace(94, 98, 20)  # Recovery
    bottom2 = np.array([98, 96, 94, 93, 92, 93, 94, 96])  # Second bottom
    recovery = np.linspace(96, 105, 30)  # Recovery
    
    prices = np.concatenate([base_trend, bottom1, middle, bottom2, recovery])
    
    # Add noise
    prices = prices + np.random.randn(len(prices)) * 0.5
    
    print(f"\nSample data: {len(prices)} price points")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Current price: ${prices[-1]:.2f}")
    
    # Detect patterns
    detector = TechnicalPatternDetector()
    patterns = detector.detect_all_patterns(prices)
    
    print(f"\nDetected {len(patterns)} patterns:\n")
    
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern.pattern}")
        print(f"   Signal: {pattern.signal}")
        print(f"   Confidence: {pattern.confidence}%")
        print(f"   Description: {pattern.description}")
        if pattern.support_level:
            print(f"   Support: ${pattern.support_level:.2f}")
        if pattern.resistance_level:
            print(f"   Resistance: ${pattern.resistance_level:.2f}")
        print()
    
    # Get aggregated signal
    df = pd.DataFrame({'close': prices})
    result = detector.get_pattern_signals(df)
    
    print("\nAggregated Signal:")
    print(f"  Signal: {result['signal']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Reasoning: {result['reasoning']}")
