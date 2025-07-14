import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List

class RiskCalculator:
    """Risk hesaplama ve pozisyon belirleme sistemi"""
    
    def __init__(self, account_balance: float):
        self.account_balance = account_balance
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float, 
                              risk_percentage: float = 2.0) -> Dict:
        """Pozisyon büyüklüğü hesaplar"""
        
        # Risk miktarı (TL cinsinden)
        risk_amount = self.account_balance * (risk_percentage / 100)
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        if risk_per_share == 0:
            return {'error': 'Stop loss fiyatı giriş fiyatına eşit olamaz'}
        
        # Hisse sayısı
        shares = int(risk_amount / risk_per_share)
        
        # Toplam yatırım miktarı
        total_investment = shares * entry_price
        
        # Portföy yüzdesi
        portfolio_percentage = (total_investment / self.account_balance) * 100
        
        return {
            'shares': shares,
            'total_investment': total_investment,
            'risk_amount': risk_amount,
            'risk_per_share': risk_per_share,
            'portfolio_percentage': portfolio_percentage,
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price
        }
    
    def calculate_stop_loss_levels(self, current_price: float, data: pd.DataFrame) -> Dict:
        """Çeşitli stop-loss seviyelerini hesaplar"""
        
        # ATR tabanlı stop-loss
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Destek/Direnç tabanlı stop-loss
        recent_lows = data['Low'].rolling(20).min().iloc[-1]
        recent_highs = data['High'].rolling(20).max().iloc[-1]
        
        # Percentage tabanlı stop-loss seviyeleri
        stop_levels = {
            'conservative_5pct': current_price * 0.95,
            'moderate_3pct': current_price * 0.97,
            'aggressive_2pct': current_price * 0.98,
            'atr_based': current_price - (2 * atr),
            'support_based': recent_lows,
            'trailing_5pct': current_price * 0.95
        }
        
        # Her stop-loss için risk/reward hesapla
        for level_name, stop_price in stop_levels.items():
            risk_amount = current_price - stop_price
            risk_percentage = (risk_amount / current_price) * 100
            stop_levels[level_name] = {
                'price': stop_price,
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage
            }
        
        return stop_levels
    
    def calculate_target_prices(self, entry_price: float, stop_loss_price: float,
                               risk_reward_ratios: List[float] = [1, 1.5, 2, 3]) -> Dict:
        """Hedef fiyatları hesaplar"""
        
        risk_amount = abs(entry_price - stop_loss_price)
        targets = {}
        
        for ratio in risk_reward_ratios:
            if entry_price > stop_loss_price:  # Long pozisyon
                target_price = entry_price + (risk_amount * ratio)
            else:  # Short pozisyon
                target_price = entry_price - (risk_amount * ratio)
            
            profit_amount = abs(target_price - entry_price)
            profit_percentage = (profit_amount / entry_price) * 100
            
            targets[f'target_{ratio}x'] = {
                'price': target_price,
                'profit_amount': profit_amount,
                'profit_percentage': profit_percentage,
                'risk_reward_ratio': ratio
            }
        
        return targets
    
    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """Portföy genelinde risk hesaplar"""
        
        total_risk = 0
        total_investment = 0
        
        for position in positions:
            position_risk = position.get('risk_amount', 0)
            position_investment = position.get('total_investment', 0)
            
            total_risk += position_risk
            total_investment += position_investment
        
        portfolio_risk_percentage = (total_risk / self.account_balance) * 100
        capital_utilization = (total_investment / self.account_balance) * 100
        
        return {
            'total_risk': total_risk,
            'total_investment': total_investment,
            'portfolio_risk_percentage': portfolio_risk_percentage,
            'capital_utilization': capital_utilization,
            'available_capital': self.account_balance - total_investment,
            'max_additional_risk': max(0, (self.account_balance * 0.1) - total_risk)  # Max %10 risk
        }
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly Criterion ile optimal pozisyon büyüklüğü"""
        
        if avg_loss == 0:
            return 0
        
        # Kelly formülü: f = (bp - q) / b
        # b = ortalama kazanç / ortalama kayıp
        # p = kazanma olasılığı
        # q = kaybetme olasılığı (1-p)
        
        b = avg_win / avg_loss
        p = win_rate / 100
        q = 1 - p
        
        kelly_percentage = (b * p - q) / b
        
        # Güvenlik için maksimum %25 ile sınırla
        kelly_percentage = max(0, min(0.25, kelly_percentage))
        
        return kelly_percentage * 100
    
    def analyze_correlation_risk(self, holdings: List[str], correlation_matrix: pd.DataFrame) -> Dict:
        """Korelasyon riskini analiz eder"""
        
        if correlation_matrix.empty or len(holdings) < 2:
            return {'correlation_risk': 'Low', 'max_correlation': 0}
        
        # Holdings arasındaki maksimum korelasyon
        max_correlation = 0
        correlated_pairs = []
        
        for i, stock1 in enumerate(holdings):
            for j, stock2 in enumerate(holdings[i+1:], i+1):
                if stock1 in correlation_matrix.index and stock2 in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[stock1, stock2])
                    if corr > max_correlation:
                        max_correlation = corr
                    
                    if corr > 0.7:  # Yüksek korelasyon
                        correlated_pairs.append((stock1, stock2, corr))
        
        # Risk seviyesi belirleme
        if max_correlation > 0.8:
            risk_level = 'Very High'
        elif max_correlation > 0.6:
            risk_level = 'High'
        elif max_correlation > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'correlation_risk': risk_level,
            'max_correlation': max_correlation,
            'correlated_pairs': correlated_pairs,
            'diversification_score': 1 - max_correlation
        } 