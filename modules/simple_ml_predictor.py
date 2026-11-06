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
    
    def safe_divide(self, numerator, denominator, default_value=1.0):
        """Güvenli bölme işlemi - sıfıra bölme ve infinity durumlarını önler"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = np.where(np.isfinite(result), result, default_value)
            return result
    
    def prepare_features(self, data: pd.DataFrame, technical_indicators: Dict) -> pd.DataFrame:
        """Basit özellik matrisi hazırlar"""
        features = pd.DataFrame(index=data.index)
        
        # Fiyat özellikleri - güvenli bölme ile
        features['price_change'] = data['Close'].pct_change()
        features['volume_change'] = data['Volume'].pct_change()
        features['high_low_ratio'] = self.safe_divide(data['High'], data['Low'], 1.0)
        
        # Hareketli ortalamalar
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['price_sma_ratio'] = self.safe_divide(data['Close'], features['sma_20'], 1.0)
        
        # Teknik indikatörler (mevcut olanlardan)
        if 'rsi' in technical_indicators:
            features['rsi'] = technical_indicators['rsi'] / 100  # Normalize
        
        for ema_name in ['ema_5', 'ema_8', 'ema_13', 'ema_21']:
            if ema_name in technical_indicators:
                ema_values = technical_indicators[ema_name]
                features[f'{ema_name}_ratio'] = self.safe_divide(data['Close'], ema_values, 1.0)
        
        # VWAP özellikleri - güvenli bölme ile
        if 'vwap' in technical_indicators:
            vwap_values = technical_indicators['vwap']
            features['vwap_ratio'] = self.safe_divide(data['Close'], vwap_values, 1.0)
            features['vwap_distance'] = self.safe_divide(
                (data['Close'] - vwap_values), data['Close'], 0.0
            )
        
        # Lag features
        features['close_lag_1'] = data['Close'].shift(1)
        features['close_lag_3'] = data['Close'].shift(3)
        features['return_lag_1'] = data['Close'].pct_change(1)
        
        # Volatilite
        features['volatility'] = data['Close'].rolling(10).std()
        
        # NaN ve infinity temizle
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().fillna(0)
        
        return features
    
    def train_quick_model(self, data: pd.DataFrame, technical_indicators: Dict) -> Dict:
        """Hızlı model eğitimi"""
        self.is_trained = False
        self.last_prediction = None
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
            if not np.isfinite([mse, mae]).all():
                raise ValueError("Model training metrics must be finite")
            
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
        self.last_prediction = None
        
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
            if not np.isfinite(predicted_return):
                raise ValueError("Predicted return must be finite")
            if not np.isfinite(current_price) or current_price <= 0:
                raise ValueError("Current price must be positive and finite")
            with np.errstate(over='ignore', invalid='ignore'):
                predicted_price = current_price * (1 + predicted_return)
                predicted_percent = predicted_return * 100
                price_change = predicted_price - current_price
            if not np.isfinite([predicted_price, predicted_percent, price_change]).all() or predicted_price <= 0:
                raise ValueError("Predicted price and derived values must be valid and finite")
            
            # Güven skoru (basit)
            confidence = min(0.8, max(0.2, 0.6 - abs(predicted_return) * 10))
            
            result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_return': predicted_percent,
                'price_change': price_change,
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
            importances = np.asarray(self.model.feature_importances_)
            feature_names = list(self.model.feature_names_in_)
            if (importances.shape != (len(feature_names),) or not feature_names
                    or len(set(feature_names)) != len(feature_names)
                    or not np.isfinite(importances).all()):
                return pd.DataFrame()
            
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return df
            
        except Exception:
            return pd.DataFrame()
