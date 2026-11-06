import pandas as pd
import numpy as np
import ta
from numbers import Integral, Real
from typing import Dict, List, Optional, Tuple
from .config import INDICATORS_CONFIG
from .pattern_recognition_advanced import AdvancedPatternRecognition

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
            'vwma_5': self._calculate_vwma,
            'vwema_5': self._calculate_vwema,
            'vwema_20': self._calculate_vwema,
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger': self._calculate_bollinger_bands,
            'stoch': self._calculate_stochastic,
            'williams_r': self._calculate_williams_r,
            'cci': self._calculate_cci,
            'supertrend': self._calculate_supertrend,
            'ott': self._calculate_ott,
            'vwap': self._calculate_vwap,
            'fvg': self._calculate_fvg,
            'order_block': self._calculate_order_block,
            'bos': self._calculate_bos,
            'fvg_ob_combo': self._calculate_fvg_ob_combo,
            'fvg_bos_combo': self._calculate_fvg_bos_combo
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
    
    def _calculate_vwma(self, indicator_name: str) -> None:
        """Hacim Ağırlıklı Hareketli Ortalama hesaplar"""
        period = INDICATORS_CONFIG[indicator_name]['period']
        
        # VWMA hesaplaması: (Close * Volume) toplamı / Volume toplamı
        typical_price_volume = self.data['Close'] * self.data['Volume']
        vwma = typical_price_volume.rolling(window=period).sum() / self.data['Volume'].rolling(window=period).sum()
        
        # NaN değerleri temizle
        vwma = vwma.fillna(self.data['Close'])
        
        self.indicators[indicator_name] = vwma
    
    @staticmethod
    def _indicator_period(value) -> int:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError("Indicator period must be a positive integer")
        return int(value)

    @staticmethod
    def _indicator_number(value, label: str, allow_zero: bool = False) -> float:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{label} must contain real numbers")
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be finite") from exc
        if not np.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
            raise ValueError(f"{label} must be finite and {'nonnegative' if allow_zero else 'positive'}")
        return number

    def _indicator_values(self, column: str, allow_zero: bool = False) -> np.ndarray:
        return np.array([self._indicator_number(value, column, allow_zero)
                         for value in self.data[column]], dtype=float)

    def _calculate_vwema(self, indicator_name: str) -> None:
        """ADR-0010 causal EMA(price*volume)/EMA(volume), first positive seed."""
        period = self._indicator_period(INDICATORS_CONFIG[indicator_name]['period'])
        close = self._indicator_values('Close')
        volume = self._indicator_values('Volume', allow_zero=True)
        try:
            alpha = 2 / (period + 1)
        except (OverflowError, ValueError) as exc:
            raise ValueError("VWEMA smoothing must be representable") from exc
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("VWEMA smoothing must be representable")
        decay = 1 - alpha
        values = np.full(len(close), np.nan, dtype=float)
        level = None
        effective_volume = 0.0
        for i, (price, weight) in enumerate(zip(close, volume)):
            if weight == 0:
                if level is not None:
                    effective_volume *= decay
                    values[i] = level
                continue
            if level is None or alpha == 1:
                level, effective_volume = price, weight
            else:
                # Normalized masses give the same N/D ratio without price*volume
                # overflow. Zero-volume bars still decay the denominator state.
                scale = max(effective_volume, weight)
                old_mass = decay * (effective_volume / scale)
                new_mass = alpha * (weight / scale)
                total_mass = old_mass + new_mass
                if not np.isfinite(total_mass) or total_mass <= 0:
                    raise ValueError("VWEMA mass must remain finite and positive")
                blend = new_mass / total_mass
                level = price if blend == 1 else level + blend * (price - level)
                effective_volume = scale * total_mass
            if not np.isfinite(level) or level <= 0 or not np.isfinite(effective_volume):
                raise ValueError("VWEMA result must remain finite and positive")
            values[i] = level
        self.indicators[indicator_name] = pd.Series(values, index=self.data.index, dtype=float)
    
    def _calculate_rsi(self, indicator_name: str) -> None:
        """RSI hesaplar - Gelişmiş pivot point ve trend çizgisi analizi ile"""
        config = INDICATORS_CONFIG[indicator_name]
        period = config['period']
        rsi_ema_length = config.get('rsi_ema_length', 66)
        pivot_point_period = config.get('pivot_point_period', 10)
        pivot_points_to_check = config.get('pivot_points_to_check', 10)
        
        # Temel RSI hesapla
        rsi = ta.momentum.rsi(self.data['Close'], window=period)
        self.indicators['rsi'] = rsi
        
        # RSI EMA hesapla
        rsi_ema = rsi.ewm(span=rsi_ema_length).mean()
        self.indicators['rsi_ema'] = rsi_ema
        
        # Pivot point analizi
        if config.get('show_pivot_points', True):
            pivot_highs, pivot_lows = self._find_rsi_pivot_points(
                rsi, pivot_point_period, pivot_points_to_check
            )
            self.indicators['rsi_pivot_highs'] = pivot_highs
            self.indicators['rsi_pivot_lows'] = pivot_lows
        
        # Trend çizgileri analizi
        if config.get('show_broken_trend_lines', True):
            trend_lines = self._calculate_rsi_trend_lines(rsi, pivot_point_period)
            self.indicators['rsi_trend_lines'] = trend_lines
    
    def _find_rsi_pivot_points(self, rsi, pivot_period, points_to_check):
        """RSI için pivot high ve low noktalarını bulur"""
        pivot_highs = pd.Series(index=rsi.index, dtype=float)
        pivot_lows = pd.Series(index=rsi.index, dtype=float)
        
        for i in range(pivot_period, len(rsi) - pivot_period):
            # Pivot High kontrolü
            current_high = rsi.iloc[i]
            is_pivot_high = True
            
            for j in range(i - pivot_period, i + pivot_period + 1):
                if j != i and not pd.isna(rsi.iloc[j]) and rsi.iloc[j] >= current_high:
                    is_pivot_high = False
                    break
            
            if is_pivot_high and not pd.isna(current_high):
                pivot_highs.iloc[i] = current_high
            
            # Pivot Low kontrolü
            current_low = rsi.iloc[i]
            is_pivot_low = True
            
            for j in range(i - pivot_period, i + pivot_period + 1):
                if j != i and not pd.isna(rsi.iloc[j]) and rsi.iloc[j] <= current_low:
                    is_pivot_low = False
                    break
            
            if is_pivot_low and not pd.isna(current_low):
                pivot_lows.iloc[i] = current_low
        
        return pivot_highs, pivot_lows
    
    def _calculate_rsi_trend_lines(self, rsi, pivot_period):
        """RSI trend çizgilerini hesaplar"""
        trend_lines = {
            'resistance_lines': [],
            'support_lines': [],
            'broken_lines': []
        }
        
        # Pivot noktalarını bul
        pivot_highs, pivot_lows = self._find_rsi_pivot_points(rsi, pivot_period, 10)
        
        # Direnç çizgileri (pivot high'lar arası)
        high_points = [(i, val) for i, val in enumerate(pivot_highs) if not pd.isna(val)]
        for i in range(len(high_points) - 1):
            for j in range(i + 1, len(high_points)):
                idx1, val1 = high_points[i]
                idx2, val2 = high_points[j]
                
                # Trend çizgisi parametreleri
                slope = (val2 - val1) / (idx2 - idx1)
                intercept = val1 - slope * idx1
                
                trend_lines['resistance_lines'].append({
                    'start_idx': idx1,
                    'end_idx': idx2,
                    'start_val': val1,
                    'end_val': val2,
                    'slope': slope,
                    'intercept': intercept
                })
        
        # Destek çizgileri (pivot low'lar arası)
        low_points = [(i, val) for i, val in enumerate(pivot_lows) if not pd.isna(val)]
        for i in range(len(low_points) - 1):
            for j in range(i + 1, len(low_points)):
                idx1, val1 = low_points[i]
                idx2, val2 = low_points[j]
                
                # Trend çizgisi parametreleri
                slope = (val2 - val1) / (idx2 - idx1)
                intercept = val1 - slope * idx1
                
                trend_lines['support_lines'].append({
                    'start_idx': idx1,
                    'end_idx': idx2,
                    'start_val': val1,
                    'end_val': val2,
                    'slope': slope,
                    'intercept': intercept
                })
        
        return trend_lines
    
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
        """ADR-0010 readiness with the existing basic-band/direction recurrence."""
        config = INDICATORS_CONFIG[indicator_name]
        period = self._indicator_period(config['period'])
        multiplier = self._indicator_number(config['multiplier'], 'Supertrend multiplier')
        high_values = self._indicator_values('High')
        low_values = self._indicator_values('Low')
        close_values = self._indicator_values('Close')
        if (high_values < low_values).any():
            raise ValueError("High must be greater than or equal to Low")
        prices = np.full(len(self.data), np.nan, dtype=float)
        directions = np.full(len(self.data), np.nan, dtype=float)
        if len(self.data) <= period:
            self.indicators['supertrend'] = pd.Series(prices, index=self.data.index, dtype=float)
            self.indicators['supertrend_trend'] = pd.Series(directions, index=self.data.index, dtype=float)
            return

        high = pd.Series(high_values, index=self.data.index)
        low = pd.Series(low_values, index=self.data.index)
        close = pd.Series(close_values, index=self.data.index)
        try:
            atr = ta.volatility.average_true_range(high, low, close, window=period)
        except Exception:
            high_low = high - low
            high_close = abs(high - close.shift())
            low_close = abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
        midpoint = low_values + (high_values - low_values) / 2
        with np.errstate(over='ignore', invalid='ignore'):
            upper = midpoint + multiplier * np.asarray(atr, dtype=float)
            lower = midpoint - multiplier * np.asarray(atr, dtype=float)
        if not np.isfinite(upper[period:]).all() or not np.isfinite(lower[period:]).all():
            raise ValueError("Supertrend bands must remain finite")
        prices[period], directions[period] = lower[period], 1.0
        for i in range(period + 1, len(prices)):
            directions[i] = -1.0 if close_values[i] <= prices[i-1] else 1.0
            prices[i] = lower[i] if directions[i] == 1 else upper[i]
        self.indicators['supertrend'] = pd.Series(prices, index=self.data.index, dtype=float)
        self.indicators['supertrend_trend'] = pd.Series(directions, index=self.data.index, dtype=float)
        
    def _calculate_ott(self, indicator_name: str) -> None:
        """Optimized Trend Tracker (OTT) hesaplar"""
        if self.data.empty:
            self.indicators['ott'] = pd.Series(index=self.data.index, dtype=float)
            self.indicators['ott_signal'] = pd.Series(index=self.data.index, dtype=str)
            self.indicators['ott_trend'] = pd.Series(index=self.data.index, dtype=float)
            return

        try:
            close = pd.to_numeric(self.data['Close'], errors='raise')
            temporal_close = (pd.api.types.is_datetime64_any_dtype(self.data['Close'])
                              or pd.api.types.is_timedelta64_dtype(self.data['Close']))
            valid_close = (not temporal_close and not pd.api.types.is_complex_dtype(close)
                           and close.notna().all() and np.isfinite(close).all()
                           and (close > 0).all())
        except (TypeError, ValueError):
            valid_close = False
        if not valid_close:
            raise ValueError("Close must be positive and finite for OTT calculation")

        config = INDICATORS_CONFIG[indicator_name]
        period = config.get('period', 2)
        percent = config.get('percent', 1.4)

        # VAR (Volatility Adjusted Average) hesaplaması
        self.data['VAR'] = self.data['Close'].ewm(span=period).mean()

        # OTT hesaplaması
        fark = self.data['VAR'] * percent / 100
        long_stop = self.data['VAR'] - fark
        short_stop = self.data['VAR'] + fark

        self.data['long_stop'] = long_stop
        self.data['short_stop'] = short_stop

        # OTT sinyal çizgisi
        ott = pd.Series(index=self.data.index, dtype=float)
        ott_signal = pd.Series(index=self.data.index, dtype=str)
        ott_trend = pd.Series(index=self.data.index, dtype=float)
        ott.iloc[0] = self.data['VAR'].iloc[0]

        for i in range(1, len(self.data)):
            if self.data['VAR'].iloc[i] > ott.iloc[i-1]:
                ott.iloc[i] = max(ott.iloc[i-1], self.data['long_stop'].iloc[i])
                ott_signal.iloc[i] = 'buy'
                ott_trend.iloc[i] = 1.0
            else:
                ott.iloc[i] = min(ott.iloc[i-1], self.data['short_stop'].iloc[i])
                ott_signal.iloc[i] = 'sell'
                ott_trend.iloc[i] = -1.0
        
        ott = ott.ffill()

        self.indicators['ott'] = ott
        self.indicators['ott_signal'] = ott_signal
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
    
    def _calculate_fvg(self, indicator_name: str) -> None:
        """Fair Value Gap (FVG) hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        threshold_percent = config['threshold_percent']
        
        pattern_analyzer = AdvancedPatternRecognition(self.data)
        fvg_data = pattern_analyzer.detect_fair_value_gaps(threshold_percent=threshold_percent)
        
        self.indicators['fvg_bullish'] = fvg_data['bullish']
        self.indicators['fvg_bearish'] = fvg_data['bearish']
    
    def _calculate_order_block(self, indicator_name: str) -> None:
        """Order Block hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        lookback = config['lookback']
        threshold_percent = config['threshold_percent']
        
        pattern_analyzer = AdvancedPatternRecognition(self.data)
        ob_data = pattern_analyzer.detect_order_blocks(lookback=lookback, threshold_percent=threshold_percent)
        
        self.indicators['ob_bullish'] = ob_data['bullish']
        self.indicators['ob_bearish'] = ob_data['bearish']
    
    def _calculate_bos(self, indicator_name: str) -> None:
        """Break of Structure (BOS) hesaplar"""
        config = INDICATORS_CONFIG[indicator_name]
        lookback = config['lookback']
        swing_threshold = config['swing_threshold']
        
        pattern_analyzer = AdvancedPatternRecognition(self.data)
        bos_data = pattern_analyzer.detect_break_of_structure(lookback=lookback, swing_threshold=swing_threshold)
        
        self.indicators['bos_bullish'] = bos_data['bullish']
        self.indicators['bos_bearish'] = bos_data['bearish']
    
    def _calculate_fvg_ob_combo(self, indicator_name: str) -> None:
        """FVG ve Order Block kombinasyonlarını hesaplar"""
        pattern_analyzer = AdvancedPatternRecognition(self.data)
        combo_data = pattern_analyzer.get_fvg_order_block_combo()
        
        self.indicators['fvg_ob_combo'] = combo_data
    
    def _calculate_fvg_bos_combo(self, indicator_name: str) -> None:
        """FVG ve Break of Structure kombinasyonlarını hesaplar"""
        pattern_analyzer = AdvancedPatternRecognition(self.data)
        combo_data = pattern_analyzer.get_fvg_bos_combo()
        
        self.indicators['fvg_bos_combo'] = combo_data

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
