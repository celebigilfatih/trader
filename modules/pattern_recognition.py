import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class PatternRecognition:
    """Mum formasyonu tanıma sistemi"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.patterns = {}
    
    def detect_doji(self, tolerance: float = 0.1) -> pd.Series:
        """Doji formasyonu tespit eder"""
        body_size = abs(self.data['Close'] - self.data['Open'])
        candle_range = self.data['High'] - self.data['Low']
        
        # Gövde boyutu, mum aralığının %10'undan küçükse Doji
        doji = body_size <= (candle_range * tolerance)
        return doji
    
    def detect_hammer(self) -> pd.Series:
        """Çekiç formasyonu tespit eder"""
        body_size = abs(self.data['Close'] - self.data['Open'])
        upper_shadow = self.data['High'] - np.maximum(self.data['Close'], self.data['Open'])
        lower_shadow = np.minimum(self.data['Close'], self.data['Open']) - self.data['Low']
        
        # Alt gölge gövdenin en az 2 katı, üst gölge minimal
        hammer = (lower_shadow >= 2 * body_size) & (upper_shadow <= 0.1 * body_size)
        return hammer
    
    def detect_shooting_star(self) -> pd.Series:
        """Kayan yıldız formasyonu tespit eder"""
        body_size = abs(self.data['Close'] - self.data['Open'])
        upper_shadow = self.data['High'] - np.maximum(self.data['Close'], self.data['Open'])
        lower_shadow = np.minimum(self.data['Close'], self.data['Open']) - self.data['Low']
        
        # Üst gölge gövdenin en az 2 katı, alt gölge minimal
        shooting_star = (upper_shadow >= 2 * body_size) & (lower_shadow <= 0.1 * body_size)
        return shooting_star
    
    def detect_engulfing_bullish(self) -> pd.Series:
        """Yükseliş saran formasyonu tespit eder"""
        prev_open = self.data['Open'].shift(1)
        prev_close = self.data['Close'].shift(1)
        
        # Önceki mum düşüş, mevcut mum yükseliş
        prev_bearish = prev_close < prev_open
        current_bullish = self.data['Close'] > self.data['Open']
        
        # Mevcut mum önceki mumu sarar
        engulfs = (self.data['Open'] < prev_close) & (self.data['Close'] > prev_open)
        
        return prev_bearish & current_bullish & engulfs
    
    def detect_engulfing_bearish(self) -> pd.Series:
        """Düşüş saran formasyonu tespit eder"""
        prev_open = self.data['Open'].shift(1)
        prev_close = self.data['Close'].shift(1)
        
        # Önceki mum yükseliş, mevcut mum düşüş
        prev_bullish = prev_close > prev_open
        current_bearish = self.data['Close'] < self.data['Open']
        
        # Mevcut mum önceki mumu sarar
        engulfs = (self.data['Open'] > prev_close) & (self.data['Close'] < prev_open)
        
        return prev_bullish & current_bearish & engulfs
    
    def detect_morning_star(self) -> pd.Series:
        """Sabah yıldızı formasyonu tespit eder"""
        # 3 mumlu formasyon
        first_bearish = (self.data['Close'].shift(2) < self.data['Open'].shift(2))
        second_small = abs(self.data['Close'].shift(1) - self.data['Open'].shift(1)) < \
                      abs(self.data['Close'].shift(2) - self.data['Open'].shift(2)) * 0.3
        third_bullish = (self.data['Close'] > self.data['Open'])
        
        # Gap'ler
        gap_down = self.data['High'].shift(1) < self.data['Low'].shift(2)
        gap_up = self.data['Low'] > self.data['High'].shift(1)
        
        return first_bearish & second_small & third_bullish & gap_down & gap_up
    
    def detect_evening_star(self) -> pd.Series:
        """Akşam yıldızı formasyonu tespit eder"""
        # 3 mumlu formasyon
        first_bullish = (self.data['Close'].shift(2) > self.data['Open'].shift(2))
        second_small = abs(self.data['Close'].shift(1) - self.data['Open'].shift(1)) < \
                      abs(self.data['Close'].shift(2) - self.data['Open'].shift(2)) * 0.3
        third_bearish = (self.data['Close'] < self.data['Open'])
        
        # Gap'ler
        gap_up = self.data['Low'].shift(1) > self.data['High'].shift(2)
        gap_down = self.data['High'] < self.data['Low'].shift(1)
        
        return first_bullish & second_small & third_bearish & gap_up & gap_down
    
    def analyze_all_patterns(self) -> Dict[str, pd.Series]:
        """Tüm formasyonları analiz eder"""
        patterns = {
            'doji': self.detect_doji(),
            'hammer': self.detect_hammer(),
            'shooting_star': self.detect_shooting_star(),
            'bullish_engulfing': self.detect_engulfing_bullish(),
            'bearish_engulfing': self.detect_engulfing_bearish(),
            'morning_star': self.detect_morning_star(),
            'evening_star': self.detect_evening_star()
        }
        
        self.patterns = patterns
        return patterns
    
    def get_latest_patterns(self, lookback: int = 5) -> Dict[str, bool]:
        """Son N günde tespit edilen formasyonlar"""
        if not self.patterns:
            self.analyze_all_patterns()
        
        latest_patterns = {}
        for pattern_name, pattern_series in self.patterns.items():
            # Son N günde formasyon var mı?
            latest_patterns[pattern_name] = pattern_series.iloc[-lookback:].any()
        
        return latest_patterns
    
    def get_pattern_signals(self) -> Dict[str, str]:
        """Formasyonların al/sat sinyalleri"""
        latest = self.get_latest_patterns()
        
        signals = {}
        bullish_patterns = ['hammer', 'bullish_engulfing', 'morning_star', 'doji']
        bearish_patterns = ['shooting_star', 'bearish_engulfing', 'evening_star']
        
        for pattern, detected in latest.items():
            if detected:
                if pattern in bullish_patterns:
                    signals[pattern] = 'BUY'
                elif pattern in bearish_patterns:
                    signals[pattern] = 'SELL'
        
        return signals 