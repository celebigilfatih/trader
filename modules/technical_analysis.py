import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
from .config import INDICATORS_CONFIG

class TechnicalAnalyzer:
    """Teknik analiz hesaplamaları yapan sınıf"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: OHLCV verileri içeren DataFrame
        """
        self.data = data.copy()
        self.indicators = {}
        self.signals = {}
        
        # Veri kontrolü
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError("Veri OHLCV formatında olmalıdır")
    
    def add_indicator(self, indicator_name: str) -> None:
        """
        Belirtilen indikatörü hesaplar ve ekler
        
        Args:
            indicator_name: İndikatör adı
        """
        if indicator_name not in INDICATORS_CONFIG:
            raise ValueError(f"Desteklenmeyen indikatör: {indicator_name}")
        
        method_map = {
            'ema_5': self._calculate_ema,
            'ema_8': self._calculate_ema,
            'ema_13': self._calculate_ema,
            'ema_21': self._calculate_ema,
            'ema_50': self._calculate_ema,
            'ema_121': self._calculate_ema,
            'ma_200': self._calculate_sma,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger': self._calculate_bollinger_bands,
            'stoch': self._calculate_stochastic,
            'williams_r': self._calculate_williams_r,
            'cci': self._calculate_cci,
            'supertrend': self._calculate_supertrend,
            'ott': self._calculate_ott,
            'vwap': self._calculate_vwap
        }
        
        if indicator_name in method_map:
            method_map[indicator_name](indicator_name)
    
    def _calculate_sma(self, indicator_name: str) -> None:
        """Basit Hareketli Ortalama hesaplar"""
        period = INDICATORS_CONFIG[indicator_name]['period']
        self.indicators[indicator_name] = self.data['Close'].rolling(window=period).mean()
    
    def _calculate_ema(self, indicator_name: str) -> None:
        """Üssel Hareketli Ortalama hesaplar"""
        period = INDICATORS_CONFIG[indicator_name]['period']
        self.indicators[indicator_name] = self.data['Close'].ewm(span=period).mean()
    
    def _calculate_rsi(self, indicator_name: str) -> None:
        """RSI hesaplar"""
        period = INDICATORS_CONFIG[indicator_name]['period']
        self.indicators['rsi'] = ta.momentum.rsi(self.data['Close'], window=period)
    
    def _calculate_macd(self, indicator_name: str) -> None:
        """MACD hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        fast = config['fast']
        slow = config['slow']
        signal = config['signal']
        
        macd_line = ta.trend.macd(self.data['Close'], window_fast=fast, window_slow=slow)
        macd_signal = ta.trend.macd_signal(self.data['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
        macd_histogram = ta.trend.macd_diff(self.data['Close'], window_fast=fast, window_slow=slow, window_sign=signal)
        
        self.indicators['macd'] = macd_line
        self.indicators['macd_signal'] = macd_signal
        self.indicators['macd_histogram'] = macd_histogram
    
    def _calculate_bollinger_bands(self, indicator_name: str) -> None:
        """Bollinger Bantları hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        period = config['period']
        std = config['std']
        
        self.indicators['bb_upper'] = ta.volatility.bollinger_hband(self.data['Close'], window=period, window_dev=std)
        self.indicators['bb_middle'] = ta.volatility.bollinger_mavg(self.data['Close'], window=period)
        self.indicators['bb_lower'] = ta.volatility.bollinger_lband(self.data['Close'], window=period, window_dev=std)
    
    def _calculate_stochastic(self, indicator_name: str) -> None:
        """Stokastik Osilatör hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        k_period = config['k_period']
        d_period = config['d_period']
        
        self.indicators['stoch_k'] = ta.momentum.stoch(
            self.data['High'], self.data['Low'], self.data['Close'], window=k_period
        )
        self.indicators['stoch_d'] = self.indicators['stoch_k'].rolling(window=d_period).mean()
    
    def _calculate_williams_r(self, indicator_name: str) -> None:
        """Williams %R hesaplar"""
        period = INDICATORS_CONFIG[indicator_name]['period']
        self.indicators['williams_r'] = ta.momentum.williams_r(
            self.data['High'], self.data['Low'], self.data['Close'], lbp=period
        )
    
    def _calculate_cci(self, indicator_name: str) -> None:
        """Emtia Kanal Endeksi hesaplar"""
        period = INDICATORS_CONFIG[indicator_name]['period']
        self.indicators['cci'] = ta.trend.cci(
            self.data['High'], self.data['Low'], self.data['Close'], window=period
        )
    
    def _calculate_supertrend(self, indicator_name: str) -> None:
        """SuperTrend indikatörünü hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        period = config['period']
        multiplier = config['multiplier']
        
        try:
            # ATR hesapla
            atr = ta.volatility.average_true_range(
                self.data['High'], self.data['Low'], self.data['Close'], window=period
            )
        except:
            # Eğer ta.volatility çalışmazsa manuel ATR hesapla
            high_low = self.data['High'] - self.data['Low']
            high_close = abs(self.data['High'] - self.data['Close'].shift())
            low_close = abs(self.data['Low'] - self.data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
        
        # HL2 (Typical Price)
        hl2 = (self.data['High'] + self.data['Low']) / 2
        
        # Upper ve Lower Band hesapla
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # SuperTrend hesapla
        supertrend = pd.Series(index=self.data.index, dtype=float)
        trend = pd.Series(index=self.data.index, dtype=int)
        
        # NaN değerleri temizle
        upper_band = upper_band.bfill() if hasattr(upper_band, 'bfill') else upper_band
        lower_band = lower_band.bfill() if hasattr(lower_band, 'bfill') else lower_band
        
        # İlk geçerli indeksi bul
        first_valid_idx = period
        if first_valid_idx >= len(self.data):
            first_valid_idx = len(self.data) - 1
        
        # İlk değerleri ayarla
        supertrend.iloc[first_valid_idx] = lower_band.iloc[first_valid_idx]
        trend.iloc[first_valid_idx] = 1
        
        for i in range(first_valid_idx + 1, len(self.data)):
            # Trend belirleme
            if pd.isna(supertrend.iloc[i-1]):
                trend.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            elif self.data['Close'].iloc[i] <= supertrend.iloc[i-1]:
                trend.iloc[i] = -1
            elif self.data['Close'].iloc[i] >= supertrend.iloc[i-1]:
                trend.iloc[i] = 1
            else:
                trend.iloc[i] = trend.iloc[i-1]
            
            # SuperTrend değeri
            if trend.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        self.indicators['supertrend'] = supertrend
        self.indicators['supertrend_trend'] = trend
        
    def _calculate_ott(self, indicator_name: str) -> None:
        """OTT (Optimized Trend Tracker) hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        period = config['period']
        percent = config['percent']
        
        # VAR hesapla (Moving Average) - pandas rolling kullan
        var = self.data['Close'].rolling(window=period).mean()
        
        # OTT hesapla
        ott = pd.Series(index=self.data.index, dtype=float)
        
        for i in range(len(self.data)):
            if i == 0:
                ott.iloc[i] = var.iloc[i]
            else:
                fark = var.iloc[i] * percent / 100
                if var.iloc[i] > ott.iloc[i-1]:
                    ott.iloc[i] = var.iloc[i] - fark
                else:
                    ott.iloc[i] = var.iloc[i] + fark
        
        self.indicators['ott'] = ott
        
        # OTT trend belirleme
        ott_trend = pd.Series(index=self.data.index, dtype=int)
        for i in range(len(self.data)):
            if self.data['Close'].iloc[i] > ott.iloc[i]:
                ott_trend.iloc[i] = 1  # Yukarı trend
            else:
                ott_trend.iloc[i] = -1  # Aşağı trend
                
        self.indicators['ott_trend'] = ott_trend
    
    def _calculate_vwap(self, indicator_name: str) -> None:
        """VWAP (Volume Weighted Average Price) hesaplar"""
        
        # Typical Price hesapla
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        
        # VWAP hesapla - kümülatif
        cumulative_tp_volume = (typical_price * self.data['Volume']).cumsum()
        cumulative_volume = self.data['Volume'].cumsum()
        
        vwap = cumulative_tp_volume / cumulative_volume
        
        # NaN değerleri temizle
        vwap = vwap.ffill().fillna(typical_price)
        
        self.indicators['vwap'] = vwap
        
        # VWAP'a göre fiyat pozisyonu
        price_position = self.data['Close'] / vwap
        self.indicators['vwap_ratio'] = price_position
    
    def calculate_support_resistance(self, lookback: int = 20) -> Tuple[float, float]:
        """
        Destek ve direnç seviyelerini hesaplar
        
        Args:
            lookback: Geriye bakış periyodu
            
        Returns:
            Tuple: (destek, direnç)
        """
        recent_data = self.data.tail(lookback)
        support = recent_data['Low'].min()
        resistance = recent_data['High'].max()
        
        return support, resistance
    
    def detect_chart_patterns(self) -> Dict[str, bool]:
        """
        Grafik desenlerini tespit eder
        
        Returns:
            Dict: Tespit edilen desenler
        """
        patterns = {
            'double_top': False,
            'double_bottom': False,
            'head_shoulders': False,
            'triangle': False,
            'flag': False
        }
        
        # Basit desen tespiti (geliştirilmeye açık)
        recent_highs = self.data['High'].tail(50)
        recent_lows = self.data['Low'].tail(50)
        
        # Double Top tespiti
        highs_peaks = self._find_peaks(recent_highs.values)
        if len(highs_peaks) >= 2:
            patterns['double_top'] = abs(recent_highs.iloc[highs_peaks[-1]] - recent_highs.iloc[highs_peaks[-2]]) < recent_highs.mean() * 0.02
        
        # Double Bottom tespiti  
        lows_valleys = self._find_valleys(recent_lows.values)
        if len(lows_valleys) >= 2:
            patterns['double_bottom'] = abs(recent_lows.iloc[lows_valleys[-1]] - recent_lows.iloc[lows_valleys[-2]]) < recent_lows.mean() * 0.02
        
        return patterns
    
    def _find_peaks(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Tepe noktalarını bulur"""
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] > data[i-j] for j in range(1, min_distance + 1)) and \
               all(data[i] > data[i+j] for j in range(1, min_distance + 1)):
                peaks.append(i)
        return peaks
    
    def _find_valleys(self, data: np.ndarray, min_distance: int = 5) -> List[int]:
        """Dip noktalarını bulur"""
        valleys = []
        for i in range(min_distance, len(data) - min_distance):
            if all(data[i] < data[i-j] for j in range(1, min_distance + 1)) and \
               all(data[i] < data[i+j] for j in range(1, min_distance + 1)):
                valleys.append(i)
        return valleys
    
    def calculate_trend_strength(self) -> Dict[str, float]:
        """
        Trend gücünü hesaplar
        
        Returns:
            Dict: Trend bilgileri
        """
        # ADX hesapla
        adx = ta.trend.adx(self.data['High'], self.data['Low'], self.data['Close'], window=14)
        
        # Fiyat trendi
        price_trend = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[-20]) / self.data['Close'].iloc[-20] * 100
        
        # Volume trendi
        volume_trend = (self.data['Volume'].tail(5).mean() - self.data['Volume'].tail(20).mean()) / self.data['Volume'].tail(20).mean() * 100
        
        return {
            'adx': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0,
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'trend_direction': 'up' if price_trend > 0 else 'down'
        }
    
    def get_latest_indicators(self) -> Dict[str, float]:
        """
        En son indikatör değerlerini döndürür
        
        Returns:
            Dict: İndikatör adı -> değer
        """
        latest_values = {}
        
        for indicator_name, values in self.indicators.items():
            if isinstance(values, pd.Series) and not values.empty:
                latest_value = values.iloc[-1]
                if not pd.isna(latest_value):
                    latest_values[indicator_name] = latest_value
        
        return latest_values
    
    def generate_summary(self) -> Dict[str, any]:
        """
        Analiz özeti oluşturur
        
        Returns:
            Dict: Analiz özeti
        """
        latest_price = self.data['Close'].iloc[-1]
        prev_price = self.data['Close'].iloc[-2]
        price_change = ((latest_price - prev_price) / prev_price) * 100
        
        support, resistance = self.calculate_support_resistance()
        trend_info = self.calculate_trend_strength()
        patterns = self.detect_chart_patterns()
        
        summary = {
            'current_price': latest_price,
            'price_change': price_change,
            'support_level': support,
            'resistance_level': resistance,
            'trend_strength': trend_info,
            'chart_patterns': patterns,
            'latest_indicators': self.get_latest_indicators(),
            'volume_spike': self.data['Volume'].iloc[-1] > self.data['Volume'].tail(20).mean() * 1.5
        }
        
        return summary 