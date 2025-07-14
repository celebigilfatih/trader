import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SimpleMLPredictor:
    """Basit ve hızlı ML fiyat tahmin sistemi"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_prediction = None
    
    def prepare_features(self, data: pd.DataFrame, technical_indicators: Dict) -> pd.DataFrame:
        """Basit özellik matrisi hazırlar"""
        features = pd.DataFrame(index=data.index)
        
        # Fiyat özellikleri
        features['price_change'] = data['Close'].pct_change()
        features['volume_change'] = data['Volume'].pct_change()
        features['high_low_ratio'] = data['High'] / data['Low']
        
        # Hareketli ortalamalar
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['price_sma_ratio'] = data['Close'] / features['sma_20']
        
        # Teknik indikatörler (mevcut olanlardan)
        if 'rsi' in technical_indicators:
            features['rsi'] = technical_indicators['rsi'] / 100  # Normalize
        
        for ema_name in ['ema_5', 'ema_8', 'ema_13', 'ema_21']:
            if ema_name in technical_indicators:
                ema_values = technical_indicators[ema_name]
                features[f'{ema_name}_ratio'] = data['Close'] / ema_values
        
        # VWAP özellikleri
        if 'vwap' in technical_indicators:
            vwap_values = technical_indicators['vwap']
            features['vwap_ratio'] = data['Close'] / vwap_values
            features['vwap_distance'] = (data['Close'] - vwap_values) / data['Close']
        
        # Lag features
        features['close_lag_1'] = data['Close'].shift(1)
        features['close_lag_3'] = data['Close'].shift(3)
        features['return_lag_1'] = data['Close'].pct_change(1)
        
        # Volatilite
        features['volatility'] = data['Close'].rolling(10).std()
        
        # NaN temizle
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def train_quick_model(self, data: pd.DataFrame, technical_indicators: Dict) -> Dict:
        """Hızlı model eğitimi"""
        try:
            # Özellik hazırlama
            features = self.prepare_features(data, technical_indicators)
            
            # Hedef değişken (1 gün sonraki getiri)
            target = data['Close'].shift(-1) / data['Close'] - 1
            
            # Geçerli veriler
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 30:
                return {'success': False, 'error': 'Yetersiz veri'}
            
            # Veri bölme
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, shuffle=False
            )
            
            # Model eğitimi
            self.model.fit(X_train, y_train)
            
            # Test tahmini
            y_pred = self.model.predict(X_test)
            
            # Performans
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            
            self.is_trained = True
            
            return {
                'success': True,
                'mse': mse,
                'mae': mae,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def predict_next_day(self, data: pd.DataFrame, technical_indicators: Dict) -> Dict:
        """Ertesi gün tahmini"""
        
        if not self.is_trained:
            train_result = self.train_quick_model(data, technical_indicators)
            if not train_result['success']:
                return {'error': train_result['error']}
        
        try:
            # Son veri noktası için özellikler
            features = self.prepare_features(data, technical_indicators)
            last_features = features.iloc[-1:].fillna(0)
            
            # Tahmin
            predicted_return = self.model.predict(last_features)[0]
            current_price = data['Close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)
            
            # Güven skoru (basit)
            confidence = min(0.8, max(0.2, 0.6 - abs(predicted_return) * 10))
            
            result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return': predicted_return * 100,
                'price_change': predicted_price - current_price,
                'confidence': confidence,
                'signal': 'BUY' if predicted_return > 0.01 else 'SELL' if predicted_return < -0.01 else 'HOLD'
            }
            
            self.last_prediction = result
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Özellik önemliliklerini döner"""
        if not self.is_trained:
            return pd.DataFrame()
        
        try:
            importances = self.model.feature_importances_
            feature_names = ['price_change', 'volume_change', 'high_low_ratio', 
                           'sma_5', 'sma_20', 'price_sma_ratio', 'rsi', 
                           'ema_5_ratio', 'ema_8_ratio', 'ema_13_ratio', 'ema_21_ratio',
                           'vwap_ratio', 'vwap_distance',
                           'close_lag_1', 'close_lag_3', 'return_lag_1', 'volatility']
            
            # Mevcut feature sayısına göre ayarla
            available_features = min(len(importances), len(feature_names))
            
            df = pd.DataFrame({
                'feature': feature_names[:available_features],
                'importance': importances[:available_features]
            }).sort_values('importance', ascending=False)
            
            return df
            
        except Exception:
            return pd.DataFrame() 