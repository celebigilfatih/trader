import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .data_fetcher import BISTDataFetcher
from .technical_analysis import TechnicalAnalyzer

class StockScreener:
    """Hisse senedi tarayıcı sistemi"""
    
    def __init__(self, symbols: Dict[str, str]):
        self.symbols = symbols
        self.data_fetcher = BISTDataFetcher()
        self.screener_results = {}
    
    def _get_period_for_interval(self, interval: str) -> str:
        """Zaman dilimine göre uygun period döndürür"""
        period_map = {
            "5m": "5d",     # 5 dakika için 5 gün
            "15m": "1mo",   # 15 dakika için 1 ay
            "30m": "1mo",   # 30 dakika için 1 ay
            "1h": "3mo",    # 1 saat için 3 ay
            "4h": "6mo",    # 4 saat için 6 ay
            "1d": "1y"      # 1 gün için 1 yıl
        }
        return period_map.get(interval, "1y")
    
    def screen_by_rsi(self, rsi_min: float = 30, rsi_max: float = 70, interval: str = "1d") -> List[Dict]:
        """RSI değerine göre hisse taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) > 20:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('rsi')
                    
                    current_rsi = analyzer.indicators['rsi'].iloc[-1]
                    if rsi_min <= current_rsi <= rsi_max:
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'rsi': current_rsi,
                            'current_price': data['Close'].iloc[-1],
                            'signal': 'Oversold' if current_rsi < 35 else 'Overbought' if current_rsi > 65 else 'Neutral',
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['rsi'])
    
    def screen_by_volume(self, volume_multiplier: float = 1.5, interval: str = "1d") -> List[Dict]:
        """Hacim artışına göre tarama"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) > 20:
                    current_volume = data['Volume'].iloc[-1]
                    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                    
                    if current_volume > avg_volume * volume_multiplier:
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_volume': current_volume,
                            'avg_volume': avg_volume,
                            'volume_ratio': current_volume / avg_volume,
                            'current_price': data['Close'].iloc[-1],
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return sorted(results, key=lambda x: x['volume_ratio'], reverse=True)
    
    def screen_by_price_breakout(self, lookback_days: int = 20, interval: str = "1d") -> List[Dict]:
        """Fiyat kırılımlarına göre tarama"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) > lookback_days:
                    current_price = data['Close'].iloc[-1]
                    resistance = data['High'].rolling(lookback_days).max().iloc[-2]
                    support = data['Low'].rolling(lookback_days).min().iloc[-2]
                    
                    if current_price > resistance:
                        signal = 'Breakout Above Resistance'
                    elif current_price < support:
                        signal = 'Breakdown Below Support'
                    else:
                        continue
                    
                    results.append({
                        'symbol': symbol,
                        'name': self.symbols[symbol],
                        'current_price': current_price,
                        'resistance': resistance,
                        'support': support,
                        'signal': signal,
                        'interval': interval
                    })
            except Exception as e:
                continue
        
        return results
    
    def screen_multi_criteria(self, criteria: Dict, interval: str = "1d") -> List[Dict]:
        """Çoklu kritere göre tarama"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) > 50:
                    analyzer = TechnicalAnalyzer(data)
                    
                    # İndikatörleri hesapla
                    analyzer.add_indicator('rsi')
                    analyzer.add_indicator('macd')
                    analyzer.add_indicator('ema_21')
                    
                    stock_data = {
                        'symbol': symbol,
                        'name': self.symbols[symbol],
                        'current_price': data['Close'].iloc[-1],
                        'rsi': analyzer.indicators['rsi'].iloc[-1],
                        'macd': analyzer.indicators['macd'].iloc[-1],
                        'ema_21': analyzer.indicators['ema_21'].iloc[-1],
                        'volume_ratio': data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1],
                        'interval': interval
                    }
                    
                    # Kriterleri kontrol et
                    passes_criteria = True
                    
                    if 'rsi_min' in criteria and stock_data['rsi'] < criteria['rsi_min']:
                        passes_criteria = False
                    if 'rsi_max' in criteria and stock_data['rsi'] > criteria['rsi_max']:
                        passes_criteria = False
                    if 'price_above_ema' in criteria and criteria['price_above_ema']:
                        if stock_data['current_price'] < stock_data['ema_21']:
                            passes_criteria = False
                    if 'min_volume_ratio' in criteria and stock_data['volume_ratio'] < criteria['min_volume_ratio']:
                        passes_criteria = False
                    
                    if passes_criteria:
                        results.append(stock_data)
                        
            except Exception as e:
                continue
        
        return results
    
    def screen_vwap_bull_signal(self, interval: str = "1d") -> List[Dict]:
        """VWAP Boğa Sinyali taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 10:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('vwap')
                    analyzer.add_indicator('rsi')
                    analyzer.add_indicator('macd')
                    
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    vwap_current = analyzer.indicators['vwap'].iloc[-1]
                    vwap_prev = analyzer.indicators['vwap'].iloc[-2]
                    
                    # VWAP Crossover kontrolü
                    if prev_price <= vwap_prev and current_price > vwap_current:
                        # Hacim artışı kontrolü
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].tail(20).mean()
                        volume_increase = current_volume > (avg_volume * 1.2)
                        
                        # RSI ve MACD onayı
                        rsi_confirm = analyzer.indicators['rsi'].iloc[-1] > 50
                        macd_confirm = analyzer.indicators['macd'].iloc[-1] > analyzer.indicators['macd'].iloc[-2]
                        
                        # Sinyal gücü
                        confirmations = sum([volume_increase, rsi_confirm, macd_confirm])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': current_price,
                            'signal': 'VWAP Bull Signal',
                            'strength': strength,
                            'volume_increase': volume_increase,
                            'rsi_confirm': rsi_confirm,
                            'macd_confirm': macd_confirm,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_golden_cross(self, interval: str = "1d") -> List[Dict]:
        """Golden Cross sinyali taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 50:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('ema_21')
                    analyzer.add_indicator('ema_50')
                    analyzer.add_indicator('rsi')
                    analyzer.add_indicator('macd')
                    
                    ema21_current = analyzer.indicators['ema_21'].iloc[-1]
                    ema21_prev = analyzer.indicators['ema_21'].iloc[-2]
                    ema50_current = analyzer.indicators['ema_50'].iloc[-1]
                    ema50_prev = analyzer.indicators['ema_50'].iloc[-2]
                    
                    # Golden Cross kontrolü
                    if ema21_prev <= ema50_prev and ema21_current > ema50_current:
                        # Hacim onayı
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume_20 = data['Volume'].tail(20).mean()
                        volume_confirm = current_volume > (avg_volume_20 * 1.3)
                        
                        # RSI ve MACD güç onayı
                        rsi_strong = analyzer.indicators['rsi'].iloc[-1] > 55
                        macd_strong = analyzer.indicators['macd'].iloc[-1] > 0
                        
                        # Sinyal gücü
                        confirmations = sum([volume_confirm, rsi_strong, macd_strong])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': data['Close'].iloc[-1],
                            'signal': 'Golden Cross',
                            'strength': strength,
                            'volume_confirm': volume_confirm,
                            'rsi_strong': rsi_strong,
                            'macd_strong': macd_strong,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_macd_bull_signal(self, interval: str = "1d") -> List[Dict]:
        """MACD Boğa Sinyali taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 26:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('macd')
                    analyzer.add_indicator('rsi')
                    
                    macd_current = analyzer.indicators['macd'].iloc[-1]
                    macd_prev = analyzer.indicators['macd'].iloc[-2]
                    macd_signal_current = analyzer.indicators['macd_signal'].iloc[-1]
                    macd_signal_prev = analyzer.indicators['macd_signal'].iloc[-2]
                    
                    # MACD Bullish Crossover
                    if macd_prev <= macd_signal_prev and macd_current > macd_signal_current:
                        # Hacim onayı
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume_15 = data['Volume'].tail(15).mean()
                        volume_confirm = current_volume > (avg_volume_15 * 1.25)
                        
                        # RSI ve fiyat trend onayı
                        rsi_confirm = analyzer.indicators['rsi'].iloc[-1] > 45
                        price_trend_confirm = data['Close'].iloc[-1] > data['Close'].iloc[-3]
                        
                        # Sinyal gücü
                        confirmations = sum([volume_confirm, rsi_confirm, price_trend_confirm])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': data['Close'].iloc[-1],
                            'signal': 'MACD Bull Signal',
                            'strength': strength,
                            'volume_confirm': volume_confirm,
                            'rsi_confirm': rsi_confirm,
                            'price_trend_confirm': price_trend_confirm,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_rsi_recovery(self, interval: str = "1d") -> List[Dict]:
        """RSI Toparlanma Sinyali taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 14:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('rsi')
                    analyzer.add_indicator('macd')
                    
                    rsi_current = analyzer.indicators['rsi'].iloc[-1]
                    rsi_prev = analyzer.indicators['rsi'].iloc[-2]
                    rsi_3_candles_ago = analyzer.indicators['rsi'].iloc[-4] if len(data) >= 4 else rsi_prev
                    
                    # RSI Oversold Recovery
                    if rsi_3_candles_ago <= 30 and rsi_current > 40 and rsi_current > rsi_prev:
                        # Hacim ve momentum onayı
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume_10 = data['Volume'].tail(10).mean()
                        volume_confirm = current_volume > avg_volume_10
                        
                        # Fiyat momentum onayı
                        price_momentum = data['Close'].iloc[-1] > data['Close'].iloc[-2]
                        
                        # MACD onayı
                        macd_confirm = analyzer.indicators['macd'].iloc[-1] > analyzer.indicators['macd'].iloc[-2]
                        
                        # Sinyal gücü
                        confirmations = sum([volume_confirm, price_momentum, macd_confirm])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': data['Close'].iloc[-1],
                            'signal': 'RSI Recovery',
                            'strength': strength,
                            'rsi_current': rsi_current,
                            'volume_confirm': volume_confirm,
                            'price_momentum': price_momentum,
                            'macd_confirm': macd_confirm,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_bollinger_breakout(self, interval: str = "1d") -> List[Dict]:
        """Bollinger Sıkışma Sinyali taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 20:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('bollinger')
                    analyzer.add_indicator('rsi')
                    
                    bb_upper = analyzer.indicators['bollinger_upper'].iloc[-1]
                    bb_lower = analyzer.indicators['bollinger_lower'].iloc[-1]
                    bb_middle = analyzer.indicators['bollinger_middle'].iloc[-1]
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2]
                    
                    # Bollinger Band Squeeze kontrolü
                    bb_width = (bb_upper - bb_lower) / bb_middle
                    if len(data) >= 6:
                        bb_width_5_ago = (analyzer.indicators['bollinger_upper'].iloc[-6] - 
                                         analyzer.indicators['bollinger_lower'].iloc[-6]) / \
                                        analyzer.indicators['bollinger_middle'].iloc[-6]
                    else:
                        bb_width_5_ago = bb_width
                    
                    # Fiyat üst banda kırılım
                    if prev_price <= bb_middle and current_price > bb_upper and bb_width < bb_width_5_ago:
                        # Hacim patlaması onayı
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume_20 = data['Volume'].tail(20).mean()
                        volume_explosion = current_volume > (avg_volume_20 * 1.5)
                        
                        # RSI destekli momentum
                        rsi_value = analyzer.indicators['rsi'].iloc[-1]
                        rsi_support = 50 < rsi_value < 80
                        
                        # Fiyat momentum onayı
                        price_momentum = (current_price - prev_price) / prev_price > 0.02
                        
                        # Sinyal gücü
                        confirmations = sum([volume_explosion, rsi_support, price_momentum])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': current_price,
                            'signal': 'Bollinger Breakout',
                            'strength': strength,
                            'volume_explosion': volume_explosion,
                            'rsi_support': rsi_support,
                            'price_momentum': price_momentum,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_higher_high_low(self, interval: str = "1d") -> List[Dict]:
        """Higher High + Higher Low Pattern taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 10:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('rsi')
                    
                    # Son 8 mum için yüksek ve alçak değerler
                    recent_highs = data['High'].tail(8)
                    recent_lows = data['Low'].tail(8)
                    
                    # Higher High kontrolü
                    first_half_high = recent_highs.iloc[:4].max()
                    second_half_high = recent_highs.iloc[4:].max()
                    higher_high = second_half_high > first_half_high
                    
                    # Higher Low kontrolü
                    first_half_low = recent_lows.iloc[:4].min()
                    second_half_low = recent_lows.iloc[4:].min()
                    higher_low = second_half_low > first_half_low
                    
                    if higher_high and higher_low:
                        # Trend gücü onayları
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].tail(10).mean()
                        volume_support = current_volume > avg_volume
                        
                        # RSI trend onayı
                        rsi_current = analyzer.indicators['rsi'].iloc[-1]
                        rsi_prev = analyzer.indicators['rsi'].iloc[-3]
                        rsi_trend = rsi_current > rsi_prev and rsi_current > 50
                        
                        # Fiyat momentum onayı
                        price_momentum = data['Close'].iloc[-1] > data['Close'].iloc[-4]
                        
                        # Sinyal gücü
                        confirmations = sum([volume_support, rsi_trend, price_momentum])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': data['Close'].iloc[-1],
                            'signal': 'Higher High + Higher Low',
                            'strength': strength,
                            'volume_support': volume_support,
                            'rsi_trend': rsi_trend,
                            'price_momentum': price_momentum,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_vwap_reversal(self, interval: str = "1d") -> List[Dict]:
        """VWAP Altında Açılır, Üstünde Kapanır Sinyali taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 5:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('vwap')
                    analyzer.add_indicator('rsi')
                    
                    vwap_current = analyzer.indicators['vwap'].iloc[-1]
                    open_price = data['Open'].iloc[-1]
                    close_price = data['Close'].iloc[-1]
                    
                    # Altında açılıp üstünde kapanma
                    if open_price < vwap_current and close_price > vwap_current:
                        # Hacim ve momentum onayları
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].tail(20).mean()
                        volume_confirm = current_volume > (avg_volume * 1.3)
                        
                        # Gün içi performans
                        daily_performance = (close_price - open_price) / open_price
                        performance_strong = daily_performance > 0.02
                        
                        # RSI momentum
                        rsi_momentum = analyzer.indicators['rsi'].iloc[-1] > 55
                        
                        # Sinyal gücü
                        confirmations = sum([volume_confirm, performance_strong, rsi_momentum])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': close_price,
                            'signal': 'VWAP Reversal',
                            'strength': strength,
                            'daily_performance': daily_performance * 100,
                            'volume_confirm': volume_confirm,
                            'performance_strong': performance_strong,
                            'rsi_momentum': rsi_momentum,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_volume_breakout(self, interval: str = "1d") -> List[Dict]:
        """Volume Spike + Yatay Direnç Kırılımı taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 20:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('rsi')
                    
                    # Son 10 mumda yatay direnç seviyesi
                    recent_highs = data['High'].tail(10)
                    resistance_level = recent_highs.quantile(0.8)
                    
                    current_price = data['Close'].iloc[-1]
                    current_volume = data['Volume'].iloc[-1]
                    avg_volume = data['Volume'].tail(20).mean()
                    
                    # Direnç kırılımı ve hacim patlaması
                    resistance_break = current_price > resistance_level
                    volume_spike = current_volume > (avg_volume * 2.0)
                    
                    if resistance_break and volume_spike:
                        # Kırılım gücü onayları
                        breakout_strength = (current_price - resistance_level) / resistance_level > 0.01
                        
                        # RSI momentum onayı
                        rsi_value = analyzer.indicators['rsi'].iloc[-1]
                        rsi_strong = 50 < rsi_value < 80
                        
                        # Trend onayı
                        trend_confirm = data['Close'].iloc[-1] > data['Close'].iloc[-5]
                        
                        # Sinyal gücü
                        confirmations = sum([breakout_strength, rsi_strong, trend_confirm])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': current_price,
                            'signal': 'Volume Breakout',
                            'strength': strength,
                            'resistance_level': resistance_level,
                            'volume_ratio': current_volume / avg_volume,
                            'breakout_strength': breakout_strength,
                            'rsi_strong': rsi_strong,
                            'trend_confirm': trend_confirm,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_gap_up_signal(self, interval: str = "1d") -> List[Dict]:
        """Gap Up + Güçlü Kapanış Sinyali taraması"""
        results = []
        period = self._get_period_for_interval(interval)
        
        for symbol in self.symbols.keys():
            try:
                data = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
                if data is not None and len(data) >= 2:
                    analyzer = TechnicalAnalyzer(data)
                    analyzer.add_indicator('rsi')
                    
                    prev_close = data['Close'].iloc[-2]
                    current_open = data['Open'].iloc[-1]
                    current_close = data['Close'].iloc[-1]
                    current_volume = data['Volume'].iloc[-1]
                    
                    # Gap up kontrolü
                    gap_percent = (current_open - prev_close) / prev_close
                    gap_up = gap_percent > 0.01
                    
                    # Güçlü kapanış
                    strong_close = (current_close - current_open) / current_open > 0.02
                    
                    if gap_up and strong_close:
                        # Hacim onayı
                        avg_volume = data['Volume'].tail(10).mean()
                        volume_confirm = current_volume > (avg_volume * 1.5)
                        
                        # Gap büyüklüğü
                        big_gap = gap_percent > 0.03
                        
                        # RSI momentum
                        rsi_momentum = analyzer.indicators['rsi'].iloc[-1] > 60
                        
                        # Sinyal gücü
                        confirmations = sum([volume_confirm, big_gap, rsi_momentum])
                        if confirmations >= 2:
                            strength = "Çok Güçlü"
                        elif confirmations == 1:
                            strength = "Güçlü"
                        else:
                            strength = "Orta"
                        
                        results.append({
                            'symbol': symbol,
                            'name': self.symbols[symbol],
                            'current_price': current_close,
                            'signal': 'Gap Up Signal',
                            'strength': strength,
                            'gap_percent': gap_percent * 100,
                            'daily_gain': ((current_close - current_open) / current_open) * 100,
                            'volume_confirm': volume_confirm,
                            'big_gap': big_gap,
                            'rsi_momentum': rsi_momentum,
                            'interval': interval
                        })
            except Exception as e:
                continue
        
        return results
    
    def screen_all_bull_signals(self, interval: str = "1d") -> Dict[str, List[Dict]]:
        """Tüm boğa sinyallerini tarar"""
        return {
            'VWAP Bull Signal': self.screen_vwap_bull_signal(interval),
            'Golden Cross': self.screen_golden_cross(interval),
            'MACD Bull Signal': self.screen_macd_bull_signal(interval),
            'RSI Recovery': self.screen_rsi_recovery(interval),
            'Bollinger Breakout': self.screen_bollinger_breakout(interval),
            'Higher High + Higher Low': self.screen_higher_high_low(interval),
            'VWAP Reversal': self.screen_vwap_reversal(interval),
            'Volume Breakout': self.screen_volume_breakout(interval),
            'Gap Up Signal': self.screen_gap_up_signal(interval)
        } 