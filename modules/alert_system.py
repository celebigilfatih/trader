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
    
    def generate_bear_signal(self, analyzer) -> Dict[str, any]:
        """
        Ayı piyasası sinyallerini üretir
        
        Args:
            analyzer: TechnicalAnalyzer objesi
            
        Returns:
            Dict: Ayı sinyali bilgileri
        """
        bear_signals = []
        bear_strength = 0
        signal_details = []
        
        latest_indicators = analyzer.get_latest_indicators()
        current_price = analyzer.data['Close'].iloc[-1]
        
        # 1. RSI Aşırı Alım (70+)
        rsi = latest_indicators.get('rsi')
        if rsi and rsi >= 70:
            bear_signals.append("RSI Aşırı Alım")
            bear_strength += 1
            signal_details.append(f"RSI: {rsi:.1f} (Aşırı Alım Seviyesi)")
        
        # 2. Fiyat MA 200'ün Altında
        ma_200 = latest_indicators.get('ma_200')
        if ma_200 and not pd.isna(ma_200) and current_price < ma_200:
            bear_signals.append("Fiyat MA200 Altında")
            bear_strength += 1.5  # Güçlü sinyal
            distance_pct = ((current_price - ma_200) / ma_200) * 100
            signal_details.append(f"MA200: ₺{ma_200:.2f} (Fiyat %{distance_pct:.1f} altında)")
        
        # 3. EMA Death Cross (21 < 50)
        ema_21 = latest_indicators.get('ema_21')
        ema_50 = latest_indicators.get('ema_50')
        if (ema_21 and ema_50 and not pd.isna(ema_21) and not pd.isna(ema_50) and ema_21 < ema_50):
            bear_signals.append("EMA Death Cross")
            bear_strength += 1.2
            signal_details.append(f"EMA21: ₺{ema_21:.2f} < EMA50: ₺{ema_50:.2f}")
            
            # Death Cross'un ne kadar güçlü olduğunu kontrol et
            cross_distance = ((ema_50 - ema_21) / ema_21) * 100
            if cross_distance > 2:  # %2'den fazla uzaklık varsa daha güçlü sinyal
                bear_strength += 0.3
                signal_details[-1] += f" (%{cross_distance:.1f} uzaklık)"
        
        # 4. MACD Negatif Bölgede
        macd = latest_indicators.get('macd')
        if macd and not pd.isna(macd) and macd < 0:
            bear_signals.append("MACD Negatif")
            bear_strength += 0.8
            signal_details.append(f"MACD: {macd:.3f} (Negatif bölgede)")
        
        # 5. SuperTrend Negatif
        if 'supertrend_trend' in analyzer.indicators:
            st_trend = analyzer.indicators['supertrend_trend'].iloc[-1]
            if st_trend == -1:
                bear_signals.append("SuperTrend Negatif")
                bear_strength += 1
                signal_details.append("SuperTrend: Aşağı yönlü trend")
        
        # 6. OTT Negatif
        if 'ott_trend' in analyzer.indicators:
            ott_trend = analyzer.indicators['ott_trend'].iloc[-1]
            if ott_trend == -1:
                bear_signals.append("OTT Negatif")
                bear_strength += 1
                signal_details.append("OTT: Aşağı yönlü trend")
        
        # 7. Volume ile Düşüş
        current_volume = analyzer.data['Volume'].iloc[-1]
        avg_volume = analyzer.data['Volume'].tail(20).mean()
        prev_price = analyzer.data['Close'].iloc[-2]
        price_change = (current_price - prev_price) / prev_price
        
        if current_volume > avg_volume * 1.5 and price_change < -0.02:
            bear_signals.append("Yüksek Volume Düşüş")
            bear_strength += 1.3
            signal_details.append(f"Volume: {current_volume:,.0f} (%{((current_volume/avg_volume-1)*100):+.0f})")
        
        # 8. Bollinger Üst Bantından Düşüş
        if 'bb_upper' in analyzer.indicators and 'bb_middle' in analyzer.indicators:
            bb_upper = analyzer.indicators['bb_upper'].iloc[-1]
            bb_middle = analyzer.indicators['bb_middle'].iloc[-1]
            
            if current_price < bb_middle and analyzer.data['Close'].iloc[-2] > bb_upper:
                bear_signals.append("Bollinger Üst Bantından Düşüş")
                bear_strength += 1.1
                signal_details.append(f"Bollinger: Üst banttan ({bb_upper:.2f}) orta banda ({bb_middle:.2f}) düşüş")
        
        # 9. Düşen Hacim ile Fiyat Düşüşü (Zayıf Alıcı İlgisi)
        if len(analyzer.data) >= 10:
            volume_trend = analyzer.data['Volume'].tail(5).mean() / analyzer.data['Volume'].tail(10).head(5).mean()
            price_trend = (current_price - analyzer.data['Close'].iloc[-6]) / analyzer.data['Close'].iloc[-6]
            
            if volume_trend < 0.8 and price_trend < -0.03:  # Hacim %20 düşmüş, fiyat %3+ düşmüş
                bear_signals.append("Zayıf Hacim ile Düşüş")
                bear_strength += 0.7
                signal_details.append(f"Hacim azalışı: %{((volume_trend-1)*100):+.1f}, Fiyat: %{(price_trend*100):+.1f}")
        
        # 10. Sürekli Düşük Kapanışlar (Lower Lows)
        if len(analyzer.data) >= 5:
            recent_lows = analyzer.data['Low'].tail(5).tolist()
            is_lower_lows = all(recent_lows[i] >= recent_lows[i+1] for i in range(len(recent_lows)-1))
            
            if is_lower_lows:
                bear_signals.append("Sürekli Düşen Dipler")
                bear_strength += 0.9
                signal_details.append(f"Son 5 günde sürekli düşen dipler: {recent_lows[-1]:.2f}")
        
        # 11. VWAP Altında İşlem
        if 'vwap' in analyzer.indicators:
            vwap = analyzer.indicators['vwap'].iloc[-1]
            if not pd.isna(vwap) and current_price < vwap * 0.98:  # VWAP'ın %2 altında
                bear_signals.append("VWAP Altında İşlem")
                bear_strength += 0.6
                vwap_distance = ((current_price - vwap) / vwap) * 100
                signal_details.append(f"VWAP: ₺{vwap:.2f} (Fiyat %{vwap_distance:.1f} altında)")
        
        # 12. Momentum Kaybı (RSI Düşüş Trendi)
        if 'rsi' in analyzer.indicators and len(analyzer.indicators['rsi']) >= 5:
            rsi_current = analyzer.indicators['rsi'].iloc[-1]
            rsi_5_days_ago = analyzer.indicators['rsi'].iloc[-6] if len(analyzer.indicators['rsi']) >= 6 else rsi_current
            
            if not pd.isna(rsi_current) and not pd.isna(rsi_5_days_ago):
                if rsi_current < rsi_5_days_ago - 10:  # RSI 10 puan düşmüş
                    bear_signals.append("RSI Momentum Kaybı")
                    bear_strength += 0.8
                    signal_details.append(f"RSI düşüşü: {rsi_5_days_ago:.1f} → {rsi_current:.1f} (-{(rsi_5_days_ago-rsi_current):.1f})")
        
        # 13. Kısa Vadeli EMA'ların Düşüş Eğilimi
        if 'ema_5' in analyzer.indicators and 'ema_8' in analyzer.indicators:
            ema_5 = latest_indicators.get('ema_5')
            ema_8 = latest_indicators.get('ema_8')
            
            if (ema_5 and ema_8 and not pd.isna(ema_5) and not pd.isna(ema_8) and 
                ema_5 < ema_8 and current_price < ema_5):
                bear_signals.append("Kısa Vadeli EMA Düşüşü")
                bear_strength += 0.7
                signal_details.append(f"EMA5: ₺{ema_5:.2f} < EMA8: ₺{ema_8:.2f}, Fiyat EMA5 altında")
        
        # Ayı Gücü Seviyesi Belirleme
        if bear_strength >= 5:
            strength_level = "GÜÇLÜ AYI"
            strength_color = "#ff4757"
        elif bear_strength >= 3:
            strength_level = "ORTA AYI"
            strength_color = "#ff6348"
        elif bear_strength >= 1:
            strength_level = "ZAYIF AYI"
            strength_color = "#ffa502"
        else:
            strength_level = "AYI YOK"
            strength_color = "#2ed573"
        
        return {
            'signals': bear_signals,
            'strength': bear_strength,
            'strength_level': strength_level,
            'strength_color': strength_color,
            'signal_count': len(bear_signals),
            'details': signal_details,
            'recommendation': self._get_bear_recommendation(bear_strength)
        }
    
    def generate_comprehensive_risk_analysis(self, analyzer) -> Dict[str, any]:
        """
        Çok boyutlu risk analizi yapar
        
        Args:
            analyzer: TechnicalAnalyzer objesi
            
        Returns:
            Dict: Kapsamlı risk analizi sonuçları
        """
        risk_factors = {}
        risk_score = 0
        recommendations = []
        
        latest_indicators = analyzer.get_latest_indicators()
        current_price = analyzer.data['Close'].iloc[-1]
        
        # 1. Volatilite Analizi
        volatility_score = 0
        price_returns = analyzer.data['Close'].pct_change().dropna()
        if len(price_returns) >= 20:
            volatility = price_returns.tail(20).std() * np.sqrt(252)  # Yıllık volatilite
            if volatility > 0.4:  # %40 üzeri yüksek volatilite
                volatility_score = 2
                risk_factors['high_volatility'] = f"Yüksek volatilite: %{volatility*100:.1f}"
            elif volatility > 0.25:  # %25-40 orta volatilite
                volatility_score = 1
                risk_factors['medium_volatility'] = f"Orta volatilite: %{volatility*100:.1f}"
            else:
                risk_factors['low_volatility'] = f"Düşük volatilite: %{volatility*100:.1f}"
        
        risk_score += volatility_score
        
        # 2. Trend Gücü Analizi
        trend_score = 0
        if 'ema_21' in latest_indicators and 'ema_50' in latest_indicators:
            ema_21 = latest_indicators['ema_21']
            ema_50 = latest_indicators['ema_50']
            
            if not pd.isna(ema_21) and not pd.isna(ema_50):
                trend_distance = abs(ema_21 - ema_50) / current_price
                if trend_distance > 0.05:  # %5 üzeri güçlü trend
                    if ema_21 < ema_50:  # Düşüş trendi
                        trend_score = 2
                        risk_factors['strong_downtrend'] = f"Güçlü düşüş trendi (EMA uzaklığı: %{trend_distance*100:.1f})"
                    else:
                        risk_factors['strong_uptrend'] = f"Güçlü yükseliş trendi (EMA uzaklığı: %{trend_distance*100:.1f})"
                elif trend_distance > 0.02:  # %2-5 orta trend
                    if ema_21 < ema_50:
                        trend_score = 1
                        risk_factors['medium_downtrend'] = f"Orta düşüş trendi (EMA uzaklığı: %{trend_distance*100:.1f})"
                    else:
                        risk_factors['medium_uptrend'] = f"Orta yükseliş trendi (EMA uzaklığı: %{trend_distance*100:.1f})"
                else:
                    risk_factors['sideways_trend'] = "Yatay seyir"
        
        risk_score += trend_score
        
        # 3. Hacim Analizi
        volume_score = 0
        current_volume = analyzer.data['Volume'].iloc[-1]
        avg_volume = analyzer.data['Volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 2.0:  # 2x üzeri hacim
            recent_change = (current_price - analyzer.data['Close'].iloc[-2]) / analyzer.data['Close'].iloc[-2]
            if recent_change < -0.02:  # %2+ düşüş ile yüksek hacim
                volume_score = 2
                risk_factors['high_volume_decline'] = f"Yüksek hacimle düşüş (%{volume_ratio*100:.0f} hacim artışı)"
            else:
                risk_factors['high_volume_rise'] = f"Yüksek hacimle yükseliş (%{volume_ratio*100:.0f} hacim artışı)"
        elif volume_ratio < 0.5:  # Yarı altı hacim
            volume_score = 1
            risk_factors['low_volume'] = f"Düşük hacim (%{volume_ratio*100:.0f} ortalama)"
        else:
            risk_factors['normal_volume'] = f"Normal hacim seviyesi (%{volume_ratio*100:.0f} ortalama)"
        
        risk_score += volume_score
        
        # 4. Support/Resistance Analizi
        support_resistance_score = 0
        try:
            support, resistance = analyzer.calculate_support_resistance()
            distance_to_support = (current_price - support) / current_price
            distance_to_resistance = (resistance - current_price) / current_price
            
            if distance_to_support < 0.02:  # Desteğe %2 yakın
                support_resistance_score = 2
                risk_factors['near_support'] = f"Destek seviyesine yakın (₺{support:.2f})"
            elif distance_to_resistance < 0.02:  # Dirence %2 yakın
                support_resistance_score = 1
                risk_factors['near_resistance'] = f"Direnç seviyesine yakın (₺{resistance:.2f})"
            else:
                risk_factors['neutral_position'] = f"Destek-direnç arası (₺{support:.2f} - ₺{resistance:.2f})"
        except:
            risk_factors['support_resistance_unknown'] = "Destek-direnç hesaplanamadı"
        
        risk_score += support_resistance_score
        
        # 5. RSI Aşırı Alım/Satım Analizi
        rsi_score = 0
        rsi = latest_indicators.get('rsi')
        if rsi and not pd.isna(rsi):
            if rsi >= 80:
                rsi_score = 2
                risk_factors['rsi_overbought'] = f"RSI aşırı alım bölgesinde ({rsi:.1f})"
            elif rsi <= 20:
                rsi_score = 1
                risk_factors['rsi_oversold'] = f"RSI aşırı satım bölgesinde ({rsi:.1f})"
            elif 60 <= rsi < 80:
                rsi_score = 0.5
                risk_factors['rsi_high'] = f"RSI yüksek seviyede ({rsi:.1f})"
            else:
                risk_factors['rsi_normal'] = f"RSI normal seviyede ({rsi:.1f})"
        
        risk_score += rsi_score
        
        # 6. MACD Momentum Analizi
        macd_score = 0
        macd = latest_indicators.get('macd')
        macd_signal = latest_indicators.get('macd_signal')
        if macd and macd_signal and not pd.isna(macd) and not pd.isna(macd_signal):
            macd_histogram = macd - macd_signal
            if macd < 0 and macd_histogram < -0.1:  # Güçlü negatif momentum
                macd_score = 1.5
                risk_factors['strong_negative_momentum'] = f"Güçlü negatif momentum (MACD: {macd:.3f})"
            elif macd > 0 and macd_histogram > 0.1:  # Güçlü pozitif momentum
                risk_factors['strong_positive_momentum'] = f"Güçlü pozitif momentum (MACD: {macd:.3f})"
            else:
                risk_factors['weak_momentum'] = f"Zayıf momentum (MACD: {macd:.3f})"
        
        risk_score += macd_score
        
        # Risk Seviyesi Belirleme
        if risk_score >= 7:
            risk_level = "YÜKSEK RİSK"
            risk_color = "#ff4757"
        elif risk_score >= 4:
            risk_level = "ORTA RİSK"
            risk_color = "#ff6348"
        elif risk_score >= 2:
            risk_level = "DÜŞÜK RİSK"
            risk_color = "#ffa502"
        else:
            risk_level = "MİNİMAL RİSK"
            risk_color = "#2ed573"
        
        # Öneriler Oluşturma
        recommendations = self._generate_risk_recommendations(risk_score, risk_factors, current_price, analyzer)
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'position_sizing': self._calculate_position_sizing(risk_score),
            'stop_loss_suggestion': self._calculate_stop_loss(current_price, analyzer, risk_score),
            'take_profit_suggestion': self._calculate_take_profit(current_price, analyzer, risk_score)
        }
    
    def _generate_risk_recommendations(self, risk_score: float, risk_factors: Dict, current_price: float, analyzer) -> List[str]:
        """Risk faktörlerine göre öneriler oluşturur"""
        recommendations = []
        
        # Risk seviyesine göre genel öneriler
        if risk_score >= 7:
            recommendations.append("🚨 YÜKSEK RİSK: Pozisyon büyüklüğünü %25'e kadar azaltın")
            recommendations.append("🛡️ Dar stop-loss kullanın (%-2 to %-3)")
            recommendations.append("⏰ Günlük takip yapın, hızlı karar almaya hazır olun")
        elif risk_score >= 4:
            recommendations.append("⚠️ ORTA RİSK: Pozisyon büyüklüğünü %50'ye sınırlayın")
            recommendations.append("🛡️ Orta seviye stop-loss kullanın (%-3 to %-5)")
            recommendations.append("📊 Günlük analiz yapın")
        elif risk_score >= 2:
            recommendations.append("✅ DÜŞÜK RİSK: Normal pozisyon büyüklüğü (%75)")
            recommendations.append("🛡️ Geniş stop-loss kullanabilirsiniz (%-5 to %-7)")
            recommendations.append("📈 Haftalık takip yeterli")
        else:
            recommendations.append("🟢 MİNİMAL RİSK: Normal işlem stratejinizi sürdürün")
            recommendations.append("📊 Rutin teknik analiz takibi")
        
        # Spesifik risk faktörlerine göre öneriler
        if 'high_volatility' in risk_factors:
            recommendations.append("🌊 Yüksek volatilite nedeniyle pozisyon büyüklüğünü azaltın")
        
        if 'strong_downtrend' in risk_factors:
            recommendations.append("📉 Güçlü düşüş trendinde short pozisyon düşünün")
        
        if 'high_volume_decline' in risk_factors:
            recommendations.append("📊 Yüksek hacimli düşüş - satış baskısına dikkat")
        
        if 'near_support' in risk_factors:
            recommendations.append("🎯 Destek seviyesi yakın - toparlanma fırsatı gözleyin")
        
        if 'near_resistance' in risk_factors:
            recommendations.append("🚧 Direnç seviyesi yakın - kar realizasyonu düşünün")
        
        if 'rsi_overbought' in risk_factors:
            recommendations.append("📈 RSI aşırı alım - düzeltme beklentisi")
        
        if 'rsi_oversold' in risk_factors:
            recommendations.append("📉 RSI aşırı satım - toparlanma fırsatı")
        
        return recommendations
    
    def _calculate_position_sizing(self, risk_score: float) -> str:
        """Risk skoruna göre pozisyon büyüklüğü önerir"""
        if risk_score >= 7:
            return "Portföyün %15-25'i (Yüksek risk)"
        elif risk_score >= 4:
            return "Portföyün %25-50'si (Orta risk)"
        elif risk_score >= 2:
            return "Portföyün %50-75'i (Düşük risk)"
        else:
            return "Portföyün %75-100'ü (Minimal risk)"
    
    def _calculate_stop_loss(self, current_price: float, analyzer, risk_score: float) -> str:
        """Risk skoruna göre stop-loss seviyesi önerir"""
        if risk_score >= 7:
            stop_loss_pct = 2  # %2
        elif risk_score >= 4:
            stop_loss_pct = 3.5  # %3.5
        elif risk_score >= 2:
            stop_loss_pct = 5  # %5
        else:
            stop_loss_pct = 7  # %7
        
        stop_loss_price = current_price * (1 - stop_loss_pct / 100)
        return f"₺{stop_loss_price:.2f} (-%{stop_loss_pct}%)"
    
    def _calculate_take_profit(self, current_price: float, analyzer, risk_score: float) -> str:
        """Risk skoruna göre take-profit seviyesi önerir"""
        # Risk/Reward oranını risk skoruna göre ayarla
        if risk_score >= 7:
            reward_ratio = 1.5  # 1:1.5 risk/reward
        elif risk_score >= 4:
            reward_ratio = 2.0  # 1:2 risk/reward
        elif risk_score >= 2:
            reward_ratio = 2.5  # 1:2.5 risk/reward
        else:
            reward_ratio = 3.0  # 1:3 risk/reward
        
        # Stop-loss yüzdesini al
        if risk_score >= 7:
            stop_loss_pct = 2
        elif risk_score >= 4:
            stop_loss_pct = 3.5
        elif risk_score >= 2:
            stop_loss_pct = 5
        else:
            stop_loss_pct = 7
        
        take_profit_pct = stop_loss_pct * reward_ratio
        take_profit_price = current_price * (1 + take_profit_pct / 100)
        return f"₺{take_profit_price:.2f} (+%{take_profit_pct:.1f}%)"
    
    def _get_bear_recommendation(self, strength: float) -> str:
        """Ayı gücüne göre basit öneri verir (eski fonksiyon - geriye dönük uyumluluk için)"""
        if strength >= 5:
            return "GÜÇLÜ SAT SİNYALİ - Pozisyonları azaltın, stop-loss kullanın"
        elif strength >= 3:
            return "ORTA SAT SİNYALİ - Dikkatli olun, risk yönetimi yapın"
        elif strength >= 1:
            return "ZAYIF SAT SİNYALİ - Gelişmeleri takip edin"
        else:
            return "AYI SİNYALİ YOK - Normal işlemler"
    
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
    
    def generate_position_recommendation(self, analyzer) -> Dict[str, any]:
        """
        Kapsamlı teknik analiz ile pozisyon önerisi üretir
        
        Args:
            analyzer: TechnicalAnalyzer objesi
            
        Returns:
            Dict: AL/SAT/TUT önerisi ve detaylı analiz
        """
        latest_indicators = analyzer.get_latest_indicators()
        current_price = analyzer.data['Close'].iloc[-1]
        
        bull_score = 0  # Boğa puanı
        bear_score = 0  # Ayı puanı
        bull_signals = []
        bear_signals = []
        technical_details = []
        
        # 1. RSI Analizi
        rsi = latest_indicators.get('rsi')
        if rsi and not pd.isna(rsi):
            if rsi <= 30:
                bull_score += 2
                bull_signals.append("RSI Aşırı Satım")
                technical_details.append(f"RSI {rsi:.1f} - Güçlü alım fırsatı")
            elif rsi <= 40:
                bull_score += 1
                bull_signals.append("RSI Düşük")
                technical_details.append(f"RSI {rsi:.1f} - Alım bölgesi")
            elif rsi >= 70:
                bear_score += 2
                bear_signals.append("RSI Aşırı Alım")
                technical_details.append(f"RSI {rsi:.1f} - Satım sinyali")
            elif rsi >= 60:
                bear_score += 1
                bear_signals.append("RSI Yüksek")
                technical_details.append(f"RSI {rsi:.1f} - Dikkat bölgesi")
            else:
                technical_details.append(f"RSI {rsi:.1f} - Nötr bölge")
        
        # 2. MACD Analizi
        macd = latest_indicators.get('macd')
        macd_signal = latest_indicators.get('macd_signal')
        if macd and macd_signal and not pd.isna(macd) and not pd.isna(macd_signal):
            macd_histogram = macd - macd_signal
            if macd > macd_signal and macd > 0:
                bull_score += 2
                bull_signals.append("MACD Pozitif Kesişim")
                technical_details.append(f"MACD ({macd:.3f}) > Signal - Güçlü boğa momentumu")
            elif macd > macd_signal:
                bull_score += 1
                bull_signals.append("MACD Yükseliş")
                technical_details.append(f"MACD ({macd:.3f}) yükseliş eğilimi")
            elif macd < macd_signal and macd < 0:
                bear_score += 2
                bear_signals.append("MACD Negatif Kesişim")
                technical_details.append(f"MACD ({macd:.3f}) < Signal - Güçlü ayı momentumu")
            elif macd < macd_signal:
                bear_score += 1
                bear_signals.append("MACD Düşüş")
                technical_details.append(f"MACD ({macd:.3f}) düşüş eğilimi")
        
        # 3. EMA Trend Analizi
        ema_21 = latest_indicators.get('ema_21')
        ema_50 = latest_indicators.get('ema_50')
        if ema_21 and ema_50 and not pd.isna(ema_21) and not pd.isna(ema_50):
            if current_price > ema_21 > ema_50:
                bull_score += 2
                bull_signals.append("Güçlü Yükseliş Trendi")
                technical_details.append(f"Fiyat > EMA21 ({ema_21:.2f}) > EMA50 ({ema_50:.2f})")
            elif current_price > ema_21:
                bull_score += 1
                bull_signals.append("Kısa Vadeli Yükseliş")
                technical_details.append(f"Fiyat > EMA21 ({ema_21:.2f})")
            elif current_price < ema_21 < ema_50:
                bear_score += 2
                bear_signals.append("Güçlü Düşüş Trendi")
                technical_details.append(f"Fiyat < EMA21 ({ema_21:.2f}) < EMA50 ({ema_50:.2f})")
            elif current_price < ema_21:
                bear_score += 1
                bear_signals.append("Kısa Vadeli Düşüş")
                technical_details.append(f"Fiyat < EMA21 ({ema_21:.2f})")
        
        # 4. Moving Average 200 Trendi
        ma_200 = latest_indicators.get('ma_200')
        if ma_200 and not pd.isna(ma_200):
            if current_price > ma_200:
                bull_score += 1.5
                bull_signals.append("Uzun Vadeli Yükseliş")
                technical_details.append(f"Fiyat > MA200 ({ma_200:.2f}) - Boğa piyasası")
            else:
                bear_score += 1.5
                bear_signals.append("Uzun Vadeli Düşüş")
                technical_details.append(f"Fiyat < MA200 ({ma_200:.2f}) - Ayı piyasası")
        
        # 5. SuperTrend Analizi
        if 'supertrend_trend' in analyzer.indicators:
            st_trend = analyzer.indicators['supertrend_trend'].iloc[-1]
            if st_trend == 1:
                bull_score += 1.5
                bull_signals.append("SuperTrend Pozitif")
                technical_details.append("SuperTrend: Yükseliş sinyali")
            elif st_trend == -1:
                bear_score += 1.5
                bear_signals.append("SuperTrend Negatif")
                technical_details.append("SuperTrend: Düşüş sinyali")
        
        # 6. OTT Analizi
        if 'ott_trend' in analyzer.indicators:
            ott_trend = analyzer.indicators['ott_trend'].iloc[-1]
            if ott_trend == 1:
                bull_score += 1
                bull_signals.append("OTT Pozitif")
                technical_details.append("OTT: Yükseliş trendi")
            elif ott_trend == -1:
                bear_score += 1
                bear_signals.append("OTT Negatif")
                technical_details.append("OTT: Düşüş trendi")
        
        # 7. Bollinger Bands Analizi
        if 'bb_upper' in analyzer.indicators and 'bb_lower' in analyzer.indicators:
            bb_upper = analyzer.indicators['bb_upper'].iloc[-1]
            bb_lower = analyzer.indicators['bb_lower'].iloc[-1]
            bb_middle = analyzer.indicators['bb_middle'].iloc[-1]
            
            if current_price <= bb_lower:
                bull_score += 1.5
                bull_signals.append("Bollinger Alt Bandında")
                technical_details.append(f"Fiyat alt bantta ({bb_lower:.2f}) - Aşırı satım")
            elif current_price >= bb_upper:
                bear_score += 1.5
                bear_signals.append("Bollinger Üst Bandında")
                technical_details.append(f"Fiyat üst bantta ({bb_upper:.2f}) - Aşırı alım")
            elif current_price > bb_middle:
                bull_score += 0.5
                technical_details.append(f"Fiyat orta bantın üstünde ({bb_middle:.2f})")
            else:
                bear_score += 0.5
                technical_details.append(f"Fiyat orta bantın altında ({bb_middle:.2f})")
        
        # 8. Hacim Analizi
        current_volume = analyzer.data['Volume'].iloc[-1]
        avg_volume = analyzer.data['Volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume
        price_change = (current_price - analyzer.data['Close'].iloc[-2]) / analyzer.data['Close'].iloc[-2]
        
        if volume_ratio > 1.5:  # Yüksek hacim
            if price_change > 0.02:  # %2+ yükseliş
                bull_score += 1
                bull_signals.append("Yüksek Hacimle Yükseliş")
                technical_details.append(f"Hacim %{volume_ratio*100:.0f} artış ile pozitif hareket")
            elif price_change < -0.02:  # %2+ düşüş
                bear_score += 1
                bear_signals.append("Yüksek Hacimle Düşüş")
                technical_details.append(f"Hacim %{volume_ratio*100:.0f} artış ile negatif hareket")
        
        # 9. VWAP Analizi
        if 'vwap' in analyzer.indicators:
            vwap = analyzer.indicators['vwap'].iloc[-1]
            if not pd.isna(vwap):
                if current_price > vwap * 1.02:  # VWAP'ın %2 üstünde
                    bull_score += 1
                    bull_signals.append("VWAP Üstünde")
                    technical_details.append(f"Fiyat VWAP ({vwap:.2f}) üstünde - Kurumsal alım")
                elif current_price < vwap * 0.98:  # VWAP'ın %2 altında
                    bear_score += 1
                    bear_signals.append("VWAP Altında")
                    technical_details.append(f"Fiyat VWAP ({vwap:.2f}) altında - Kurumsal satım")
        
        # Karar Algoritması
        total_score = bull_score - bear_score
        
        if total_score >= 4:
            recommendation = "GÜÇLÜ AL"
            recommendation_color = "#00ff00"
            position_strength = "GÜÇLÜ"
        elif total_score >= 2:
            recommendation = "AL"
            recommendation_color = "#32cd32"
            position_strength = "ORTA"
        elif total_score >= 1:
            recommendation = "ZAYIF AL"
            recommendation_color = "#90ee90"
            position_strength = "ZAYIF"
        elif total_score <= -4:
            recommendation = "GÜÇLÜ SAT"
            recommendation_color = "#ff0000"
            position_strength = "GÜÇLÜ"
        elif total_score <= -2:
            recommendation = "SAT"
            recommendation_color = "#ff4500"
            position_strength = "ORTA"
        elif total_score <= -1:
            recommendation = "ZAYIF SAT"
            recommendation_color = "#ff6347"
            position_strength = "ZAYIF"
        else:
            recommendation = "TUT"
            recommendation_color = "#ffa500"
            position_strength = "NÖTR"
        
        # Pozisyon büyüklüğü önerisi
        if "GÜÇLÜ" in recommendation:
            position_size = "Portföyün %75-100'ü"
        elif "AL" in recommendation and "ZAYIF" not in recommendation:
            position_size = "Portföyün %50-75'i"
        elif "SAT" in recommendation and "ZAYIF" not in recommendation:
            position_size = "Pozisyonun %75-100'ünü sat"
        elif "ZAYIF" in recommendation:
            position_size = "Portföyün %25-50'si"
        else:
            position_size = "Mevcut pozisyonu koru"
        
        # Risk uyarıları
        risk_warnings = []
        if bear_score > 5:
            risk_warnings.append("⚠️ Yüksek düşüş riski - Dikkatli olun")
        if bull_score > 5 and bear_score > 3:
            risk_warnings.append("⚠️ Karışık sinyaller - Aşamalı pozisyon alın")
        if volume_ratio < 0.5:
            risk_warnings.append("⚠️ Düşük hacim - Breakout bekleyin")
        
        return {
            'recommendation': recommendation,
            'position_strength': position_strength,
            'recommendation_color': recommendation_color,
            'bull_score': bull_score,
            'bear_score': bear_score,
            'total_score': total_score,
            'bull_signals': bull_signals,
            'bear_signals': bear_signals,
            'technical_details': technical_details,
            'position_size': position_size,
            'risk_warnings': risk_warnings,
            'confidence': min(abs(total_score) * 10, 100)  # Güven skoru %0-100
        } 