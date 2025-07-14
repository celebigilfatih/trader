import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
from .config import ALERT_CONFIG, INDICATORS_CONFIG

# Email imports - isteğe bağlı
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

class AlertSystem:
    """Al-Sat sinyalleri ve alert sistemi"""
    
    def __init__(self):
        self.alert_history = []
        self.last_alerts = {}
        
    def generate_signal(self, analyzer) -> str:
        """
        Teknik analiz sonuçlarına göre al-sat sinyali üretir
        
        Args:
            analyzer: TechnicalAnalyzer objesi
            
        Returns:
            str: "AL", "SAT" veya "BEKLE"
        """
        signals = []
        latest_indicators = analyzer.get_latest_indicators()
        
        # RSI sinyali
        rsi_signal = self._rsi_signal(latest_indicators.get('rsi'))
        if rsi_signal:
            signals.append(rsi_signal)
        
        # MACD sinyali
        macd_signal = self._macd_signal(analyzer)
        if macd_signal:
            signals.append(macd_signal)
        
        # Moving Average sinyali
        ma_signal = self._moving_average_signal(analyzer)
        if ma_signal:
            signals.append(ma_signal)
        
        # SuperTrend sinyali
        st_signal = self._supertrend_signal(analyzer)
        if st_signal:
            signals.append(st_signal)
        
        # OTT sinyali
        ott_signal = self._ott_signal(analyzer)
        if ott_signal:
            signals.append(ott_signal)
        
        # Bollinger Bands sinyali
        bb_signal = self._bollinger_signal(analyzer)
        if bb_signal:
            signals.append(bb_signal)
        
        # Volume sinyali
        volume_signal = self._volume_signal(analyzer)
        if volume_signal:
            signals.append(volume_signal)
        
        # Sinyalleri birleştir
        return self._combine_signals(signals)
    
    def _rsi_signal(self, rsi_value: Optional[float]) -> Optional[str]:
        """RSI'ya göre sinyal üretir"""
        if rsi_value is None:
            return None
        
        config = INDICATORS_CONFIG['rsi']
        
        if rsi_value <= config['oversold']:
            return "AL"  # Aşırı satılmış
        elif rsi_value >= config['overbought']:
            return "SAT"  # Aşırı alınmış
        
        return None
    
    def _macd_signal(self, analyzer) -> Optional[str]:
        """MACD'ye göre sinyal üretir"""
        if 'macd' not in analyzer.indicators or 'macd_signal' not in analyzer.indicators:
            return None
        
        macd_line = analyzer.indicators['macd'].dropna()
        macd_signal = analyzer.indicators['macd_signal'].dropna()
        
        if len(macd_line) < 2 or len(macd_signal) < 2:
            return None
        
        # MACD çizgisinin sinyal çizgisini kesme durumu
        current_macd = macd_line.iloc[-1]
        current_signal = macd_signal.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = macd_signal.iloc[-2]
        
        # Yukarı kesim (AL sinyali)
        if prev_macd <= prev_signal and current_macd > current_signal:
            return "AL"
        # Aşağı kesim (SAT sinyali)
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return "SAT"
        
        return None
    
    def _moving_average_signal(self, analyzer) -> Optional[str]:
        """Hareketli ortalama kesişimlerine göre sinyal üretir"""
        if 'ema_21' not in analyzer.indicators or 'ema_50' not in analyzer.indicators:
            return None
        
        ema_21 = analyzer.indicators['ema_21'].dropna()
        ema_50 = analyzer.indicators['ema_50'].dropna()
        
        if len(ema_21) < 2 or len(ema_50) < 2:
            return None
        
        current_21 = ema_21.iloc[-1]
        current_50 = ema_50.iloc[-1]
        prev_21 = ema_21.iloc[-2]
        prev_50 = ema_50.iloc[-2]
        
        # Golden Cross (EMA 21, EMA 50'yi yukarı keser)
        if prev_21 <= prev_50 and current_21 > current_50:
            return "AL"
        # Death Cross (EMA 21, EMA 50'yi aşağı keser)
        elif prev_21 >= prev_50 and current_21 < current_50:
            return "SAT"
        
        return None
    
    def _supertrend_signal(self, analyzer) -> Optional[str]:
        """SuperTrend'e göre sinyal üretir"""
        if 'supertrend_trend' not in analyzer.indicators:
            return None
        
        trend = analyzer.indicators['supertrend_trend'].dropna()
        
        if len(trend) < 2:
            return None
        
        current_trend = trend.iloc[-1]
        prev_trend = trend.iloc[-2]
        
        # Trend değişimi
        if prev_trend == -1 and current_trend == 1:
            return "AL"  # Aşağı trendden yukarı trende geçiş
        elif prev_trend == 1 and current_trend == -1:
            return "SAT"  # Yukarı trendden aşağı trende geçiş
        
        return None
    
    def _ott_signal(self, analyzer) -> Optional[str]:
        """OTT'ye göre sinyal üretir"""
        if 'ott_trend' not in analyzer.indicators:
            return None
        
        trend = analyzer.indicators['ott_trend'].dropna()
        
        if len(trend) < 2:
            return None
        
        current_trend = trend.iloc[-1]
        prev_trend = trend.iloc[-2]
        
        # Trend değişimi
        if prev_trend == -1 and current_trend == 1:
            return "AL"  # Aşağı trendden yukarı trende geçiş
        elif prev_trend == 1 and current_trend == -1:
            return "SAT"  # Yukarı trendden aşağı trende geçiş
        
        return None
    
    def _bollinger_signal(self, analyzer) -> Optional[str]:
        """Bollinger Bantlarına göre sinyal üretir"""
        if 'bb_upper' not in analyzer.indicators or 'bb_lower' not in analyzer.indicators:
            return None
        
        current_price = analyzer.data['Close'].iloc[-1]
        bb_upper = analyzer.indicators['bb_upper'].iloc[-1]
        bb_lower = analyzer.indicators['bb_lower'].iloc[-1]
        
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            return None
        
        # Fiyat alt banda yaklaşırsa AL
        if current_price <= bb_lower * 1.02:  # %2 tolerans
            return "AL"
        # Fiyat üst banda yaklaşırsa SAT
        elif current_price >= bb_upper * 0.98:  # %2 tolerans
            return "SAT"
        
        return None
    
    def _volume_signal(self, analyzer) -> Optional[str]:
        """Volume analizine göre sinyal üretir"""
        current_volume = analyzer.data['Volume'].iloc[-1]
        avg_volume = analyzer.data['Volume'].tail(20).mean()
        
        current_price = analyzer.data['Close'].iloc[-1]
        prev_price = analyzer.data['Close'].iloc[-2]
        
        price_change = (current_price - prev_price) / prev_price
        
        # Yüksek volume ile fiyat artışı
        if current_volume > avg_volume * ALERT_CONFIG['volume_spike_multiplier'] and price_change > 0.02:
            return "AL"
        # Yüksek volume ile fiyat düşüşü
        elif current_volume > avg_volume * ALERT_CONFIG['volume_spike_multiplier'] and price_change < -0.02:
            return "SAT"
        
        return None
    
    def _combine_signals(self, signals: List[str]) -> str:
        """Birden fazla sinyali birleştirir"""
        if not signals:
            return "BEKLE"
        
        al_count = signals.count("AL")
        sat_count = signals.count("SAT")
        
        # Çoğunluk kuralı
        if al_count > sat_count:
            return "AL"
        elif sat_count > al_count:
            return "SAT"
        else:
            return "BEKLE"
    
    def check_price_alerts(self, analyzer, target_price: float = None, stop_loss: float = None) -> List[Dict]:
        """
        Fiyat alertlerini kontrol eder
        
        Args:
            analyzer: TechnicalAnalyzer objesi
            target_price: Hedef fiyat
            stop_loss: Zarar durdurma fiyatı
            
        Returns:
            List[Dict]: Tetiklenen alertler
        """
        alerts = []
        current_price = analyzer.data['Close'].iloc[-1]
        
        if target_price and current_price >= target_price:
            alerts.append({
                'type': 'price_target',
                'message': f'Hedef fiyat {target_price:.2f} TL ulaşıldı! Güncel: {current_price:.2f} TL',
                'timestamp': datetime.now(),
                'price': current_price
            })
        
        if stop_loss and current_price <= stop_loss:
            alerts.append({
                'type': 'stop_loss',
                'message': f'Stop loss {stop_loss:.2f} TL tetiklendi! Güncel: {current_price:.2f} TL',
                'timestamp': datetime.now(),
                'price': current_price
            })
        
        return alerts
    
    def check_technical_alerts(self, analyzer) -> List[Dict]:
        """
        Teknik indikatör alertlerini kontrol eder
        
        Args:
            analyzer: TechnicalAnalyzer objesi
            
        Returns:
            List[Dict]: Tetiklenen alertler
        """
        alerts = []
        latest_indicators = analyzer.get_latest_indicators()
        
        # RSI alertleri
        rsi = latest_indicators.get('rsi')
        if rsi:
            if rsi <= ALERT_CONFIG['rsi_oversold']:
                alerts.append({
                    'type': 'rsi_oversold',
                    'message': f'RSI aşırı satılmış seviyede: {rsi:.2f}',
                    'timestamp': datetime.now(),
                    'value': rsi
                })
            elif rsi >= ALERT_CONFIG['rsi_overbought']:
                alerts.append({
                    'type': 'rsi_overbought',
                    'message': f'RSI aşırı alınmış seviyede: {rsi:.2f}',
                    'timestamp': datetime.now(),
                    'value': rsi
                })
        
        # Volume spike alertleri
        current_volume = analyzer.data['Volume'].iloc[-1]
        avg_volume = analyzer.data['Volume'].tail(20).mean()
        
        if current_volume > avg_volume * ALERT_CONFIG['volume_spike_multiplier']:
            alerts.append({
                'type': 'volume_spike',
                'message': f'Volume artışı tespit edildi: {current_volume:,.0f} (Ort: {avg_volume:,.0f})',
                'timestamp': datetime.now(),
                'value': current_volume / avg_volume
            })
        
        # Fiyat değişim alertleri
        current_price = analyzer.data['Close'].iloc[-1]
        prev_price = analyzer.data['Close'].iloc[-2]
        price_change_pct = abs((current_price - prev_price) / prev_price * 100)
        
        if price_change_pct > ALERT_CONFIG['price_change_threshold']:
            direction = "artış" if current_price > prev_price else "düşüş"
            alerts.append({
                'type': 'price_change',
                'message': f'Büyük fiyat {direction}: %{price_change_pct:.2f}',
                'timestamp': datetime.now(),
                'value': price_change_pct
            })
        
        return alerts
    
    def send_email_alert(self, alert: Dict, recipient_email: str, smtp_config: Dict) -> bool:
        """
        Email alert gönderir
        
        Args:
            alert: Alert bilgileri
            recipient_email: Alıcı email
            smtp_config: SMTP ayarları
            
        Returns:
            bool: Başarılı ise True
        """
        if not EMAIL_AVAILABLE:
            print("Email modülü kullanılamıyor. Email alertleri devre dışı.")
            return False
            
        try:
            msg = MimeMultipart()
            msg['From'] = smtp_config['sender_email']
            msg['To'] = recipient_email
            msg['Subject'] = f"BIST Alert: {alert['type']}"
            
            body = f"""
            Alert Türü: {alert['type']}
            Mesaj: {alert['message']}
            Zaman: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
            server.starttls()
            server.login(smtp_config['sender_email'], smtp_config['password'])
            
            text = msg.as_string()
            server.sendmail(smtp_config['sender_email'], recipient_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email gönderme hatası: {str(e)}")
            return False
    
    def save_alert_history(self, alerts: List[Dict], filename: str = "alert_history.csv") -> None:
        """Alert geçmişini kaydet"""
        try:
            df = pd.DataFrame(alerts)
            
            # Mevcut dosya varsa ekle, yoksa oluştur
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(filename, index=False)
            
        except Exception as e:
            print(f"Alert geçmişi kaydetme hatası: {str(e)}")
    
    def get_signal_strength(self, analyzer) -> Dict[str, float]:
        """
        Sinyal gücünü hesaplar
        
        Args:
            analyzer: TechnicalAnalyzer objesi
            
        Returns:
            Dict: Sinyal gücü bilgileri
        """
        latest_indicators = analyzer.get_latest_indicators()
        strength = {
            'overall': 0,
            'trend': 0,
            'momentum': 0,
            'volume': 0
        }
        
        # RSI momentum
        rsi = latest_indicators.get('rsi', 50)
        if rsi <= 30:
            strength['momentum'] += 0.8  # Güçlü al sinyali
        elif rsi >= 70:
            strength['momentum'] -= 0.8  # Güçlü sat sinyali
        elif 40 <= rsi <= 60:
            strength['momentum'] += 0.2  # Nötr
        
        # Trend analizi
        if 'sma_20' in analyzer.indicators and 'sma_50' in analyzer.indicators:
            sma_20 = analyzer.indicators['sma_20'].iloc[-1]
            sma_50 = analyzer.indicators['sma_50'].iloc[-1]
            current_price = analyzer.data['Close'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                strength['trend'] += 0.6
            elif current_price < sma_20 < sma_50:
                strength['trend'] -= 0.6
        
        # Volume analizi
        current_volume = analyzer.data['Volume'].iloc[-1]
        avg_volume = analyzer.data['Volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 1.5:
            strength['volume'] += 0.4
        elif volume_ratio < 0.5:
            strength['volume'] -= 0.2
        
        # Genel güç
        strength['overall'] = (strength['trend'] + strength['momentum'] + strength['volume']) / 3
        
        return strength 