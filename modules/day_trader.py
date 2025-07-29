import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .technical_analysis import TechnicalAnalyzer
from .alert_system import AlertSystem
from .data_fetcher import BISTDataFetcher
from .config import BIST_SYMBOLS, INDICATORS_CONFIG

class DayTrader:
    """Günlük Trading (Day Trading) ve Scalping için özel modül"""
    
    def __init__(self):
        self.data_fetcher = BISTDataFetcher()
        self.alert_system = AlertSystem()
        self.active_positions = {}
        self.trading_history = []
        
    def scan_intraday_opportunities(self, timeframe: str = "5m") -> List[Dict]:
        """
        Günlük trading fırsatlarını tarar
        
        Args:
            timeframe: Zaman dilimi ("1m", "5m", "15m", "1h")
            
        Returns:
            List[Dict]: Trading fırsatları listesi
        """
        opportunities = []
        
        for symbol, name in BIST_SYMBOLS.items():
            try:
                # Veri al
                data = self.data_fetcher.get_stock_data(symbol, period="5d", interval=timeframe)
                if data is None or len(data) < 50:
                    continue
                
                # Teknik analiz
                analyzer = TechnicalAnalyzer(data)
                
                # Entry/Exit noktalarını hesapla
                entry_exit = self._calculate_entry_exit_points(data, analyzer)
                
                if entry_exit['signal'] != 'HOLD':
                    opportunities.append({
                        'symbol': symbol,
                        'name': name,
                        'signal': entry_exit['signal'],
                        'entry_price': entry_exit['entry_price'],
                        'stop_loss': entry_exit['stop_loss'],
                        'take_profit': entry_exit['take_profit'],
                        'risk_reward': entry_exit['risk_reward'],
                        'confidence': entry_exit['confidence'],
                        'timeframe': timeframe,
                        'current_price': data['Close'].iloc[-1],
                        'volume_ratio': self._calculate_volume_ratio(data),
                        'atr_percent': self._calculate_atr_percent(data),
                    })
                    
            except Exception as e:
                continue
                
        # Confidence'a göre sırala
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities[:20]  # En iyi 20 fırsat
    
    def generate_scalping_signals(self) -> List[Dict]:
        """
        Ultra kısa vadeli scalping sinyalleri üretir (2-10 dakika)
        
        Returns:
            List[Dict]: Scalping sinyalleri
        """
        scalping_signals = []
        
        for symbol, name in list(BIST_SYMBOLS.items())[:30]:  # İlk 30 hisse için
            try:
                # 1 dakikalık veri al
                data = self.data_fetcher.get_stock_data(symbol, period="1d", interval="1m")
                if data is None or len(data) < 20:
                    continue
                
                signal = self._analyze_scalping_opportunity(data, symbol, name)
                if signal['action'] != 'WAIT':
                    scalping_signals.append(signal)
                    
            except Exception as e:
                continue
        
        # Signal strength'e göre sırala
        scalping_signals.sort(key=lambda x: x['strength'], reverse=True)
        return scalping_signals[:10]  # En güçlü 10 sinyal
    
    def _calculate_entry_exit_points(self, data: pd.DataFrame, analyzer: TechnicalAnalyzer) -> Dict:
        """Entry ve exit noktalarını hesaplar"""
        
        # Teknik indikatörler - doğru API kullan
        analyzer.add_indicator('rsi')
        analyzer.add_indicator('macd')
        analyzer.add_indicator('ema_5')
        analyzer.add_indicator('ema_21')
        analyzer.add_indicator('vwap')
        analyzer.add_indicator('bollinger')
        
        rsi = analyzer.indicators.get('rsi', pd.Series())
        macd_line = analyzer.indicators.get('macd', pd.Series())
        macd_signal = analyzer.indicators.get('macd_signal', pd.Series())
        macd_histogram = analyzer.indicators.get('macd_histogram', pd.Series())
        ema_5 = analyzer.indicators.get('ema_5', pd.Series())
        ema_21 = analyzer.indicators.get('ema_21', pd.Series())
        vwap = analyzer.indicators.get('vwap', pd.Series())
        bb_upper = analyzer.indicators.get('bb_upper', pd.Series())
        bb_middle = analyzer.indicators.get('bb_middle', pd.Series())
        bb_lower = analyzer.indicators.get('bb_lower', pd.Series())
        
        current_price = data['Close'].iloc[-1]
        atr = self._calculate_atr(data, 14)
        
        # Sinyal hesaplama
        signal_score = 0
        signal_reasons = []
        
        # RSI sinyalleri (güvenli kontrol)
        if len(rsi) > 1 and not rsi.empty:
            if rsi.iloc[-1] < 30 and rsi.iloc[-2] >= 30:
                signal_score += 3
                signal_reasons.append("RSI Oversold Exit")
            elif rsi.iloc[-1] > 70 and rsi.iloc[-2] <= 70:
                signal_score -= 3
                signal_reasons.append("RSI Overbought Entry")
            
        # MACD sinyalleri (güvenli kontrol)
        if len(macd_line) > 1 and len(macd_signal) > 1 and not macd_line.empty and not macd_signal.empty:
            if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_line.iloc[-2] <= macd_signal.iloc[-2]:
                signal_score += 2
                signal_reasons.append("MACD Bullish Cross")
            elif macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_line.iloc[-2] >= macd_signal.iloc[-2]:
                signal_score -= 2
                signal_reasons.append("MACD Bearish Cross")
                
        # EMA sinyalleri (güvenli kontrol)
        if len(ema_5) > 1 and len(ema_21) > 1 and not ema_5.empty and not ema_21.empty:
            if ema_5.iloc[-1] > ema_21.iloc[-1] and ema_5.iloc[-2] <= ema_21.iloc[-2]:
                signal_score += 2
                signal_reasons.append("EMA5 > EMA21 Cross")
            elif ema_5.iloc[-1] < ema_21.iloc[-1] and ema_5.iloc[-2] >= ema_21.iloc[-2]:
                signal_score -= 2
                signal_reasons.append("EMA5 < EMA21 Cross")
                
        # VWAP sinyalleri (güvenli kontrol)
        if len(vwap) > 0 and not vwap.empty:
            if current_price > vwap.iloc[-1]:
                signal_score += 1
                signal_reasons.append("Above VWAP")
            else:
                signal_score -= 1
                signal_reasons.append("Below VWAP")
                
        # Bollinger Bands sinyalleri (güvenli kontrol)
        if len(bb_lower) > 0 and len(bb_upper) > 0 and not bb_lower.empty and not bb_upper.empty:
            if current_price <= bb_lower.iloc[-1]:
                signal_score += 2
                signal_reasons.append("BB Lower Touch")
            elif current_price >= bb_upper.iloc[-1]:
                signal_score -= 2
                signal_reasons.append("BB Upper Touch")
        
        # Sinyal belirleme (daha duyarlı threshold)
        if signal_score >= 2:  # 3'ten 2'ye düşürdük
            signal = 'BUY'
            entry_price = current_price * 1.002  # %0.2 yukarıda entry
            stop_loss = current_price - (atr * 1.5)  # 1.5 ATR stop
            take_profit = current_price + (atr * 3)   # 3 ATR profit (1:2 R:R)
        elif signal_score <= -2:  # -3'ten -2'ye düşürdük
            signal = 'SELL'
            entry_price = current_price * 0.998  # %0.2 aşağıda entry
            stop_loss = current_price + (atr * 1.5)  # 1.5 ATR stop
            take_profit = current_price - (atr * 3)   # 3 ATR profit
        else:
            signal = 'HOLD'
            entry_price = current_price
            stop_loss = current_price
            take_profit = current_price
            
        # Risk/Reward hesaplama
        if signal != 'HOLD':
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
            
        # Confidence hesaplama (0-100)
        confidence = min(abs(signal_score) * 15, 100)
        
        return {
            'signal': signal,
            'entry_price': round(entry_price, 3),
            'stop_loss': round(stop_loss, 3),
            'take_profit': round(take_profit, 3),
            'risk_reward': round(risk_reward, 2),
            'confidence': confidence,
            'reasons': signal_reasons,
            'signal_score': signal_score
        }
    
    def _analyze_scalping_opportunity(self, data: pd.DataFrame, symbol: str, name: str) -> Dict:
        """Scalping fırsatını analiz eder"""
        
        if len(data) < 20:
            return {'action': 'WAIT', 'strength': 0}
        
        analyzer = TechnicalAnalyzer(data)
        
        # Kısa vadeli indikatörler - doğru API kullan
        analyzer.add_indicator('ema_5')  # En yakın EMA'ları kullan
        analyzer.add_indicator('ema_8')
        analyzer.add_indicator('rsi')
        
        ema_3 = analyzer.indicators.get('ema_5', pd.Series())  # 3 yerine 5 kullan
        ema_8 = analyzer.indicators.get('ema_8', pd.Series())
        rsi = analyzer.indicators.get('rsi', pd.Series())
        
        current_price = data['Close'].iloc[-1]
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        
        # Scalping koşulları
        strength = 0
        action = 'WAIT'
        reasons = []
        
        # Volume koşulu (en önemli)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.3:  # 1.5'ten 1.3'e düşürdük
            strength += 3
            reasons.append(f"Yüksek Hacim ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.0:  # Normal hacim için de puan verelim
            strength += 1
            reasons.append(f"Normal+ Hacim ({volume_ratio:.1f}x)")
        
        # EMA momentum (güvenli kontrol)
        if len(ema_3) > 1 and len(ema_8) > 1 and not ema_3.empty and not ema_8.empty:
            if ema_3.iloc[-1] > ema_8.iloc[-1] and ema_3.iloc[-2] <= ema_8.iloc[-2]:
                strength += 2
                action = 'BUY'
                reasons.append("EMA5>EMA8 Cross")
            elif ema_3.iloc[-1] < ema_8.iloc[-1] and ema_3.iloc[-2] >= ema_8.iloc[-2]:
                strength += 2
                action = 'SELL'
                reasons.append("EMA5<EMA8 Cross")
        
        # RSI momentum (güvenli kontrol)
        if len(rsi) > 1 and not rsi.empty:
            if 30 < rsi.iloc[-1] < 40 and rsi.iloc[-1] > rsi.iloc[-2]:
                strength += 1
                reasons.append("RSI Recovery")
            elif 60 < rsi.iloc[-1] < 70 and rsi.iloc[-1] < rsi.iloc[-2]:
                strength += 1
                reasons.append("RSI Decline")
        
        # Fiyat momentum (son 3 mum)
        price_momentum = (current_price - data['Close'].iloc[-4]) / data['Close'].iloc[-4] * 100
        if abs(price_momentum) > 0.3:  # %0.3'den fazla hareket (daha duyarlı)
            strength += 1
            reasons.append(f"Price Momentum {price_momentum:.2f}%")
        
        # En az 2 puan olmalı scalping için (daha duyarlı)
        if strength < 2:
            action = 'WAIT'
            
        return {
            'symbol': symbol,
            'name': name,
            'action': action,
            'strength': strength,
            'current_price': current_price,
            'volume_ratio': round(volume_ratio, 2),
            'price_momentum': round(price_momentum, 2),
            'reasons': reasons,
            'target_duration': '2-10 dakika',
            'recommended_size': '1-2% portföy'
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Average True Range hesaplar"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean().iloc[-1]
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """Hacim oranını hesaplar"""
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        return current_volume / avg_volume if avg_volume > 0 else 1
    
    def _calculate_atr_percent(self, data: pd.DataFrame) -> float:
        """ATR'yi yüzde olarak hesaplar"""
        atr = self._calculate_atr(data)
        current_price = data['Close'].iloc[-1]
        return (atr / current_price) * 100 if current_price > 0 else 0
    
    def get_active_positions(self) -> List[Dict]:
        """Aktif pozisyonları döndürür"""
        return list(self.active_positions.values())
    
    def add_position(self, symbol: str, entry_price: float, position_type: str, 
                    stop_loss: float, take_profit: float, size: float):
        """Yeni pozisyon ekler"""
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'type': position_type,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': size,
            'entry_time': datetime.now(),
            'status': 'ACTIVE'
        }
        self.active_positions[symbol] = position
        
    def close_position(self, symbol: str, exit_price: float, reason: str):
        """Pozisyon kapatır"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            position['exit_price'] = exit_price
            position['exit_time'] = datetime.now()
            position['close_reason'] = reason
            position['status'] = 'CLOSED'
            
            # Kar/Zarar hesapla
            if position['type'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - exit_price) * position['size']
                
            position['pnl'] = pnl
            
            # Trading history'ye ekle
            self.trading_history.append(position.copy())
            
            # Aktif pozisyonlardan çıkar
            del self.active_positions[symbol] 