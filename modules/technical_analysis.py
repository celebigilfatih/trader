import pandas as pd
import numpy as np
import ta
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
    
    def _calculate_vwema(self, indicator_name: str) -> None:
        """Hacim Ağırlıklı Üssel Hareketli Ortalama hesaplar"""
        period = INDICATORS_CONFIG[indicator_name]['period']
        
        # VWEMA hesaplaması: EMA'yı hacim ile ağırlıklandırma
        # İlk olarak, close price'ı volume ile çarpıyoruz
        volume_weighted_price = self.data['Close'] * self.data['Volume']
        
        # EMA hesaplama katsayısı
        multiplier = 2 / (period + 1)
        
        # İlk değer olarak VWMA kullan
        initial_vwma = (volume_weighted_price.iloc[:period].sum() / 
                       self.data['Volume'].iloc[:period].sum()) if len(self.data) >= period else self.data['Close'].iloc[0]
        
        vwema_values = []
        
        # EMA hesaplama
        for i in range(len(self.data)):
            if i == 0:
                vwema_values.append(initial_vwma)
            else:
                current_volume = self.data['Volume'].iloc[i]
                if current_volume > 0:
                    current_vw_price = volume_weighted_price.iloc[i] / current_volume
                    current_vwema = (current_vw_price * multiplier) + (vwema_values[-1] * (1 - multiplier))
                else:
                    current_vwema = vwema_values[-1]
                vwema_values.append(current_vwema)
        
        # Pandas Series olarak oluştur
        import pandas as pd
        vwema_series = pd.Series(vwema_values, index=self.data.index)
        
        # NaN değerleri temizle
        vwema_series = vwema_series.fillna(self.data['Close'])
        
        self.indicators[indicator_name] = vwema_series
    
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
        """OTT (Optimized Trend Tracker) hesaplar - Pine Script versiyonuna göre"""
        config = INDICATORS_CONFIG[indicator_name]
        period = config['period']
        percent = config['percent']
        
        # Pine Script'teki VAR (Variable Moving Average) hesaplama
        def calculate_var(src, length):
            """Variable Moving Average hesaplar"""
            alpha = 2 / (length + 1)
            
            # Up/Down movement hesapla
            up_moves = np.where(src > src.shift(1), src - src.shift(1), 0)
            down_moves = np.where(src < src.shift(1), src.shift(1) - src, 0)
            
            # 9 günlük toplamlar
            ud_sum = pd.Series(up_moves).rolling(window=9).sum()
            dd_sum = pd.Series(down_moves).rolling(window=9).sum()
            
            # CMO (Chande Momentum Oscillator)
            cmo = (ud_sum - dd_sum) / (ud_sum + dd_sum)
            cmo = cmo.fillna(0)
            
            # VAR hesaplama
            var = pd.Series(index=src.index, dtype=float)
            var.iloc[0] = src.iloc[0]
            
            for i in range(1, len(src)):
                abs_cmo = abs(cmo.iloc[i])
                var.iloc[i] = alpha * abs_cmo * src.iloc[i] + (1 - alpha * abs_cmo) * var.iloc[i-1]
            
            return var
        
        # Pine Script'teki WWMA (Welles Wilder Moving Average) hesaplama
        def calculate_wwma(src, length):
            """Welles Wilder Moving Average hesaplar"""
            alpha = 1 / length
            wwma = pd.Series(index=src.index, dtype=float)
            wwma.iloc[0] = src.iloc[0]
            
            for i in range(1, len(src)):
                wwma.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * wwma.iloc[i-1]
            
            return wwma
        
        # Pine Script'teki ZLEMA (Zero Lag Exponential Moving Average) hesaplama
        def calculate_zlema(src, length):
            """Zero Lag Exponential Moving Average hesaplar"""
            lag = length // 2 if length % 2 == 0 else (length - 1) // 2
            ema_data = src + (src - src.shift(lag))
            zlema = ema_data.ewm(span=length).mean()
            return zlema
        
        # Pine Script'teki TSF (Time Series Forecast) hesaplama
        def calculate_tsf(src, length):
            """Time Series Forecast hesaplar"""
            tsf = pd.Series(index=src.index, dtype=float)
            
            for i in range(length-1, len(src)):
                # Linear regression hesapla
                x = np.arange(length)
                y = src.iloc[i-length+1:i+1].values
                
                if len(y) == length:
                    slope, intercept = np.polyfit(x, y, 1)
                    tsf.iloc[i] = slope * (length - 1) + intercept + slope
                else:
                    tsf.iloc[i] = src.iloc[i]
            
            return tsf
        
        # Moving Average türünü seç (konfigürasyondan al)
        ma_type = config.get('ma_type', 'VAR')  # Varsayılan VAR
        
        if ma_type == "VAR":
            ma = calculate_var(self.data['Close'], period)
        elif ma_type == "WWMA":
            ma = calculate_wwma(self.data['Close'], period)
        elif ma_type == "ZLEMA":
            ma = calculate_zlema(self.data['Close'], period)
        elif ma_type == "TSF":
            ma = calculate_tsf(self.data['Close'], period)
        elif ma_type == "SMA":
            ma = self.data['Close'].rolling(window=period).mean()
        elif ma_type == "EMA":
            ma = self.data['Close'].ewm(span=period).mean()
        elif ma_type == "WMA":
            # Weighted Moving Average
            weights = np.arange(1, period + 1)
            ma = self.data['Close'].rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        elif ma_type == "TMA":
            # Triangular Moving Average
            half_period = period // 2
            ma = self.data['Close'].rolling(window=half_period).mean().rolling(window=half_period + 1).mean()
        else:
            # Varsayılan olarak VAR kullan
            ma = calculate_var(self.data['Close'], period)
        
        # OTT hesaplama - Pine Script algoritmasına göre
        fark = ma * percent * 0.01
        
        # Long Stop hesaplama
        long_stop = ma - fark
        long_stop_prev = long_stop.shift(1).fillna(long_stop)
        long_stop = np.where(ma > long_stop_prev, np.maximum(long_stop, long_stop_prev), long_stop)
        
        # Short Stop hesaplama
        short_stop = ma + fark
        short_stop_prev = short_stop.shift(1).fillna(short_stop)
        short_stop = np.where(ma < short_stop_prev, np.minimum(short_stop, short_stop_prev), short_stop)
        
        # Direction hesaplama
        dir_series = pd.Series(index=self.data.index, dtype=int)
        dir_series.iloc[0] = 1  # Başlangıç yönü
        
        for i in range(1, len(self.data)):
            prev_dir = dir_series.iloc[i-1]
            ma_val = ma.iloc[i]
            short_stop_prev_val = short_stop_prev.iloc[i]
            long_stop_prev_val = long_stop_prev.iloc[i]
            
            if prev_dir == -1 and ma_val > short_stop_prev_val:
                dir_series.iloc[i] = 1
            elif prev_dir == 1 and ma_val < long_stop_prev_val:
                dir_series.iloc[i] = -1
            else:
                dir_series.iloc[i] = prev_dir
        
        # MT (Moving Trend) hesaplama
        mt = np.where(dir_series == 1, long_stop, short_stop)
        
        # OTT hesaplama
        ott = np.where(ma > mt, mt * (200 + percent) / 200, mt * (200 - percent) / 200)
        
        # OTT'yi 2 periyot gecikmeli olarak hesapla (Pine Script'teki gibi)
        ott_shifted = pd.Series(ott).shift(2)
        
        self.indicators['ott'] = ott_shifted
        
        # OTT trend belirleme
        ott_trend = pd.Series(index=self.data.index, dtype=int)
        for i in range(len(self.data)):
            if pd.notna(ott_shifted.iloc[i]) and self.data['Close'].iloc[i] > ott_shifted.iloc[i]:
                ott_trend.iloc[i] = 1  # Yukarı trend
            elif pd.notna(ott_shifted.iloc[i]) and self.data['Close'].iloc[i] < ott_shifted.iloc[i]:
                ott_trend.iloc[i] = -1  # Aşağı trend
            else:
                ott_trend.iloc[i] = 0  # Nötr
                
        self.indicators['ott_trend'] = ott_trend
        
        # OTT renk değişimi sinyalleri
        ott_color_change = pd.Series(index=self.data.index, dtype=bool)
        for i in range(2, len(self.data)):
            if pd.notna(ott_shifted.iloc[i]) and pd.notna(ott_shifted.iloc[i-1]):
                ott_color_change.iloc[i] = ott_shifted.iloc[i] > ott_shifted.iloc[i-1]
        
        self.indicators['ott_color_change'] = ott_color_change
        
        # OTT sinyalleri (Pine Script versiyonuna göre)
        # Support Line Crossing Signals
        ma_cross_ott = pd.Series(index=self.data.index, dtype=bool)
        ma_cross_ott_prev = pd.Series(index=self.data.index, dtype=bool)
        
        for i in range(1, len(self.data)):
            if pd.notna(ma.iloc[i]) and pd.notna(ott_shifted.iloc[i]) and pd.notna(ma.iloc[i-1]) and pd.notna(ott_shifted.iloc[i-1]):
                ma_cross_ott.iloc[i] = ma.iloc[i] > ott_shifted.iloc[i] and ma.iloc[i-1] <= ott_shifted.iloc[i-1]  # Crossover
                ma_cross_ott_prev.iloc[i] = ma.iloc[i] < ott_shifted.iloc[i] and ma.iloc[i-1] >= ott_shifted.iloc[i-1]  # Crossunder
        
        self.indicators['ott_buy_signal'] = ma_cross_ott  # Support Line BUY Signal
        self.indicators['ott_sell_signal'] = ma_cross_ott_prev  # Support Line SELL Signal
        
        # Price Crossing OTT Signals
        price_cross_ott = pd.Series(index=self.data.index, dtype=bool)
        price_cross_ott_prev = pd.Series(index=self.data.index, dtype=bool)
        
        for i in range(1, len(self.data)):
            if pd.notna(self.data['Close'].iloc[i]) and pd.notna(ott_shifted.iloc[i]) and pd.notna(self.data['Close'].iloc[i-1]) and pd.notna(ott_shifted.iloc[i-1]):
                price_cross_ott.iloc[i] = self.data['Close'].iloc[i] > ott_shifted.iloc[i] and self.data['Close'].iloc[i-1] <= ott_shifted.iloc[i-1]  # Price Crossover
                price_cross_ott_prev.iloc[i] = self.data['Close'].iloc[i] < ott_shifted.iloc[i] and self.data['Close'].iloc[i-1] >= ott_shifted.iloc[i-1]  # Price Crossunder
        
        self.indicators['ott_price_buy_signal'] = price_cross_ott  # Price BUY Signal
        self.indicators['ott_price_sell_signal'] = price_cross_ott_prev  # Price SELL Signal
    
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