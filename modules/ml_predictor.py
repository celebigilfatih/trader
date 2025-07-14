import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    """Machine Learning tabanlı fiyat tahmin sistemi"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'svr': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.trained_models = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame, technical_indicators: Dict) -> pd.DataFrame:
        """Teknik indikatörlerden özellik matrisi hazırlar"""
        features = pd.DataFrame(index=data.index)
        
        # Fiyat tabanlı özellikler
        features['price_change'] = data['Close'].pct_change()
        features['high_low_ratio'] = data['High'] / data['Low']
        features['open_close_ratio'] = data['Open'] / data['Close']
        features['volume_change'] = data['Volume'].pct_change()
        
        # Volatilite özellikleri
        features['volatility_5'] = data['Close'].rolling(5).std()
        features['volatility_20'] = data['Close'].rolling(20).std()
        
        # Trend özellikleri
        features['trend_5'] = data['Close'].rolling(5).mean() / data['Close']
        features['trend_20'] = data['Close'].rolling(20).mean() / data['Close']
        
        # Teknik indikatör özellikleri
        for indicator_name, values in technical_indicators.items():
            if isinstance(values, pd.Series) and len(values) == len(data):
                # RSI normalizasyonu
                if 'rsi' in indicator_name:
                    features[f'{indicator_name}_normalized'] = values / 100
                
                # MACD özellikleri
                elif 'macd' in indicator_name:
                    features[indicator_name] = values
                    if indicator_name == 'macd':
                        features['macd_signal_diff'] = values - technical_indicators.get('macd_signal', 0)
                
                # EMA özellikleri
                elif 'ema' in indicator_name:
                    features[f'{indicator_name}_ratio'] = data['Close'] / values
                    features[f'{indicator_name}_distance'] = (data['Close'] - values) / data['Close']
                
                # SuperTrend
                elif 'supertrend' in indicator_name:
                    if indicator_name == 'supertrend_trend':
                        features['supertrend_signal'] = values
                    else:
                        features['supertrend_distance'] = (data['Close'] - values) / data['Close']
                
                # Bollinger Bands
                elif 'bb_' in indicator_name:
                    features[indicator_name] = values
                    if indicator_name == 'bb_middle':
                        features['bb_position'] = (data['Close'] - values) / values
        
        # Lag özellikleri (geçmiş fiyat bilgileri)
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = data['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = data['Volume'].shift(lag)
            features[f'return_lag_{lag}'] = data['Close'].pct_change(lag)
        
        # Moving averages ratios
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_ratio'] = features['sma_5'] / features['sma_20']
        
        # Volume indicators
        features['volume_sma_5'] = data['Volume'].rolling(5).mean()
        features['volume_sma_20'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma_20']
        
        # Price position in range
        features['price_position'] = (data['Close'] - data['Low'].rolling(20).min()) / (data['High'].rolling(20).max() - data['Low'].rolling(20).min())
        
        # NaN değerleri temizle
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def create_target_variable(self, data: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        """Hedef değişken oluşturur (gelecekteki fiyat değişimi)"""
        future_return = data['Close'].shift(-prediction_horizon) / data['Close'] - 1
        return future_return.fillna(0)
    
    def train_models(self, data: pd.DataFrame, technical_indicators: Dict, 
                    prediction_horizon: int = 1, test_size: float = 0.2) -> Dict:
        """Modelleri eğitir ve performans metriklerini döner"""
        
        # Özellik ve hedef değişken hazırlama
        features = self.prepare_features(data, technical_indicators)
        target = self.create_target_variable(data, prediction_horizon)
        
        # Geçerli veri noktalarını filtrele
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features_clean = features[valid_idx]
        target_clean = target[valid_idx]
        
        if len(features_clean) < 50:
            raise ValueError("Yeterli veri yok. En az 50 geçerli veri noktası gerekli.")
        
        # Özellik adlarını sakla
        self.feature_names = features_clean.columns.tolist()
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            features_clean, target_clean, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Özellikleri ölçeklendir
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hedef değişkeni ölçeklendir
        y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
        
        # Modelleri eğit ve değerlendir
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Modeli eğit
                if model_name == 'svr':
                    model.fit(X_train_scaled, y_train_scaled)
                    y_pred_scaled = model.predict(X_test_scaled)
                    y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Performans metrikleri
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation skoru
                if model_name != 'svr':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
                    cv_score = -cv_scores.mean()
                else:
                    cv_score = mse
                
                results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'cv_score': cv_score,
                    'predictions': y_pred,
                    'actual': y_test.values
                }
                
                # Eğitilmiş modeli sakla
                self.trained_models[model_name] = model
                
            except Exception as e:
                print(f"Model {model_name} eğitiminde hata: {str(e)}")
                continue
        
        self.is_trained = True
        return results
    
    def predict_price(self, data: pd.DataFrame, technical_indicators: Dict, 
                     prediction_horizon: int = 1) -> Dict:
        """Gelecek fiyat tahminleri yapar"""
        
        if not self.is_trained:
            raise ValueError("Modeller henüz eğitilmemiş. Önce train_models() çağırın.")
        
        # Son veri noktası için özellikler hazırla
        features = self.prepare_features(data, technical_indicators)
        last_features = features.iloc[-1:][self.feature_names]
        
        # NaN kontrolü
        if last_features.isna().any().any():
            last_features = last_features.fillna(method='ffill').fillna(0)
        
        current_price = data['Close'].iloc[-1]
        predictions = {}
        
        for model_name, model in self.trained_models.items():
            try:
                if model_name == 'svr':
                    # SVR için ölçeklendirilmiş veri kullan
                    last_features_scaled = self.scaler.transform(last_features)
                    pred_scaled = model.predict(last_features_scaled)[0]
                    predicted_return = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
                else:
                    predicted_return = model.predict(last_features)[0]
                
                predicted_price = current_price * (1 + predicted_return)
                
                predictions[model_name] = {
                    'predicted_return': predicted_return,
                    'predicted_price': predicted_price,
                    'current_price': current_price,
                    'price_change': predicted_price - current_price,
                    'price_change_percent': predicted_return * 100
                }
                
            except Exception as e:
                print(f"Model {model_name} tahmininde hata: {str(e)}")
                continue
        
        # Ensemble tahmini (ortalama)
        if predictions:
            avg_return = np.mean([p['predicted_return'] for p in predictions.values()])
            avg_price = current_price * (1 + avg_return)
            
            predictions['ensemble'] = {
                'predicted_return': avg_return,
                'predicted_price': avg_price,
                'current_price': current_price,
                'price_change': avg_price - current_price,
                'price_change_percent': avg_return * 100
            }
        
        return predictions
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> pd.DataFrame:
        """Özellik önemliliklerini döner"""
        
        if not self.is_trained or model_name not in self.trained_models:
            return pd.DataFrame()
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()
    
    def get_model_confidence(self, predictions: Dict) -> Dict:
        """Model güven skorları hesaplar"""
        confidence_scores = {}
        
        for model_name, pred in predictions.items():
            if model_name == 'ensemble':
                continue
                
            # Basit güven skoru (tahmin edilen değişimin mutlak değeri ile ters orantılı)
            abs_change = abs(pred['predicted_return'])
            confidence = max(0, min(1, 1 - abs_change * 10))  # 0-1 arası normalize et
            
            confidence_scores[model_name] = {
                'confidence': confidence,
                'prediction_strength': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
            }
        
        return confidence_scores
    
    def save_models(self, filepath: str) -> None:
        """Eğitilmiş modelleri dosyaya kaydeder"""
        model_data = {
            'models': self.trained_models,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath: str) -> None:
        """Kaydedilmiş modelleri yükler"""
        model_data = joblib.load(filepath)
        self.trained_models = model_data['models']
        self.scaler = model_data['scaler']
        self.target_scaler = model_data['target_scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained'] 