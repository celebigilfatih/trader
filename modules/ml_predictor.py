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
    
    def safe_divide(self, numerator, denominator, default_value=0):
        """Güvenli bölme işlemi - sıfıra bölme ve infinity durumlarını önler"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = numerator / denominator
            result = np.where(np.isfinite(result), result, default_value)
            return result
    
    def clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Feature matrisini temizler - infinity, NaN ve aşırı değerleri düzeltir"""
        # Infinity değerleri temizle
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # pct_change'den gelen infinity değerleri yakalayıp temizle
        pct_cols = [col for col in features.columns if 'change' in col or 'return' in col]
        for col in pct_cols:
            features[col] = features[col].replace([np.inf, -np.inf], 0)
            # Aşırı büyük percentage değerleri clip et (-100% ile +1000% arası)
            features[col] = features[col].clip(-1, 10)
        
        # NaN değerleri forward fill ile doldur, sonra 0 ile doldur
        features = features.ffill().fillna(0)
        
        # Aşırı büyük değerleri clamp et (percentile tabanlı)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in pct_cols:  # Percentage column'ları zaten clip ettik
                q99 = features[col].quantile(0.99)
                q01 = features[col].quantile(0.01)
                if q99 != q01 and np.isfinite(q99) and np.isfinite(q01):  # Sıfır variance ve finite kontrolü
                    features[col] = features[col].clip(lower=q01, upper=q99)
        
        # Son kontrolle tüm infinity değerleri temizle
        features = features.replace([np.inf, -np.inf], 0)
        
        # Hala NaN varsa 0 ile doldur
        features = features.fillna(0)
        
        return features
    
    def diagnostic_features(self, features: pd.DataFrame) -> Dict:
        """Feature matrisindeki problemleri teşhis eder"""
        diagnostics = {}
        
        # Infinity kontrolü
        inf_cols = []
        for col in features.columns:
            if np.isinf(features[col]).any():
                inf_cols.append(col)
        
        # NaN kontrolü
        nan_cols = []
        for col in features.columns:
            if features[col].isna().any():
                nan_cols.append(col)
        
        # Aşırı büyük değer kontrolü
        large_cols = []
        for col in features.columns:
            if (np.abs(features[col]) > 1e10).any():
                large_cols.append(col)
        
        diagnostics['infinity_columns'] = inf_cols
        diagnostics['nan_columns'] = nan_cols
        diagnostics['large_value_columns'] = large_cols
        diagnostics['total_features'] = len(features.columns)
        diagnostics['total_rows'] = len(features)
        
        return diagnostics
        
    def train_models(self, data: pd.DataFrame, technical_indicators: Dict, 
                    prediction_horizon: int = 1, test_size: float = 0.2) -> Dict:
        """Modelleri eğitir ve performans metriklerini döner"""
        
        try:
            # Özellik ve hedef değişken hazırlama
            features = self.prepare_features(data, technical_indicators)
            target = self.create_target_variable(data, prediction_horizon)
            
            # Feature diagnostics (debug için)
            diagnostics = self.diagnostic_features(features)
            if diagnostics['infinity_columns'] or diagnostics['nan_columns']:
                print(f"⚠️ Feature sorunları tespit edildi: {diagnostics}")
            
            # Ek güvenlik: Son bir kez temizle
            features = self.clean_features(features)
            
            # Geçerli veri noktalarını filtrele
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features_clean = features[valid_idx]
            target_clean = target[valid_idx]
            
            if len(features_clean) < 50:
                return {"error": "Yeterli veri yok. En az 50 geçerli veri noktası gerekli.", 
                       "available_data": len(features_clean)}
            
            # Özellik adlarını sakla
            self.feature_names = features_clean.columns.tolist()
            
            # Final infinity kontrolü
            if np.isinf(features_clean.values).any() or np.isnan(features_clean.values).any():
                # Son çare: problematik satırları çıkar
                finite_mask = np.isfinite(features_clean.values).all(axis=1)
                features_clean = features_clean[finite_mask]
                target_clean = target_clean[finite_mask]
                
                if len(features_clean) < 50:
                    return {"error": "Veri temizleme sonrası yeterli veri kalmadı.", 
                           "remaining_data": len(features_clean)}
            
            # Veriyi eğitim ve test setlerine ayır
            X_train, X_test, y_train, y_test = train_test_split(
                features_clean, target_clean, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Final check before scaling
            if np.isinf(X_train.values).any() or np.isnan(X_train.values).any():
                return {"error": "X_train matrisinde hala infinite/NaN değerler var"}
            
            # Özellikleri ölçeklendir
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Hedef değişkeni ölçeklendir
            y_train_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_test_scaled = self.target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
            
            # Modelleri eğit
            results = {}
            for model_name, model in self.models.items():
                try:
                    # Model eğitimi
                    model.fit(X_train_scaled, y_train_scaled)
                    self.trained_models[model_name] = model
                    
                    # Tahmin yap
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    # Ters ölçeklendirme
                    y_pred_train_original = self.target_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
                    y_pred_test_original = self.target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                    
                    # Performans metrikleri
                    train_score = r2_score(y_train, y_pred_train_original)
                    test_score = r2_score(y_test, y_pred_test_original)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_original))
                    mae = mean_absolute_error(y_test, y_pred_test_original)
                    
                    results[model_name] = {
                        'train_score': train_score,
                        'test_score': test_score,
                        'rmse': rmse,
                        'mae': mae
                    }
                    
                except Exception as model_error:
                    results[model_name] = {'error': str(model_error)}
            
            # Genel sonuçlar
            successful_models = [name for name, result in results.items() if 'error' not in result]
            if successful_models:
                avg_train_score = np.mean([results[name]['train_score'] for name in successful_models])
                avg_test_score = np.mean([results[name]['test_score'] for name in successful_models])
                avg_rmse = np.mean([results[name]['rmse'] for name in successful_models])
                avg_mae = np.mean([results[name]['mae'] for name in successful_models])
                
                self.is_trained = True
                
                return {
                    'train_score': avg_train_score,
                    'test_score': avg_test_score,
                    'rmse': avg_rmse,
                    'mae': avg_mae,
                    'models': results,
                    'successful_models': successful_models,
                    'feature_count': len(self.feature_names),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            else:
                return {"error": "Hiçbir model başarıyla eğitilemedi", "model_errors": results}
                
        except Exception as e:
            return {"error": f"Model eğitim hatası: {str(e)}"}
    
    def prepare_features(self, data: pd.DataFrame, technical_indicators: Dict) -> pd.DataFrame:
        """Teknik indikatörlerden özellik matrisi hazırlar"""
        features = pd.DataFrame(index=data.index)
        
        # Güvenli bölme kullanarak fiyat tabanlı özellikler
        features['price_change'] = data['Close'].pct_change()
        features['high_low_ratio'] = self.safe_divide(data['High'], data['Low'], 1.0)
        features['open_close_ratio'] = self.safe_divide(data['Open'], data['Close'], 1.0)
        features['volume_change'] = data['Volume'].pct_change()
        
        # Volatilite özellikleri
        features['volatility_5'] = data['Close'].rolling(5).std()
        features['volatility_20'] = data['Close'].rolling(20).std()
        
        # Güvenli trend özellikleri
        sma_5 = data['Close'].rolling(5).mean()
        sma_20 = data['Close'].rolling(20).mean()
        features['trend_5'] = self.safe_divide(sma_5, data['Close'], 1.0)
        features['trend_20'] = self.safe_divide(sma_20, data['Close'], 1.0)
        
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
                        macd_signal = technical_indicators.get('macd_signal', pd.Series(0, index=values.index))
                        features['macd_signal_diff'] = values - macd_signal
                
                # EMA özellikleri - güvenli bölme
                elif 'ema' in indicator_name:
                    features[f'{indicator_name}_ratio'] = self.safe_divide(data['Close'], values, 1.0)
                    features[f'{indicator_name}_distance'] = self.safe_divide(
                        (data['Close'] - values), data['Close'], 0.0
                    )
                
                # SuperTrend
                elif 'supertrend' in indicator_name:
                    if indicator_name == 'supertrend_trend':
                        features['supertrend_signal'] = values
                    else:
                        features['supertrend_distance'] = self.safe_divide(
                            (data['Close'] - values), data['Close'], 0.0
                        )
                
                # Bollinger Bands - güvenli bölme
                elif 'bb_' in indicator_name:
                    features[indicator_name] = values
                    if indicator_name == 'bb_middle':
                        features['bb_position'] = self.safe_divide(
                            (data['Close'] - values), values, 0.0
                        )
        
        # Lag özellikleri (geçmiş fiyat bilgileri)
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = data['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = data['Volume'].shift(lag)
            features[f'return_lag_{lag}'] = data['Close'].pct_change(lag)
        
        # Moving averages ratios - güvenli bölme
        features['sma_5'] = sma_5
        features['sma_20'] = sma_20
        features['sma_ratio'] = self.safe_divide(features['sma_5'], features['sma_20'], 1.0)
        
        # Volume indicators - güvenli bölme
        volume_sma_5 = data['Volume'].rolling(5).mean()
        volume_sma_20 = data['Volume'].rolling(20).mean()
        features['volume_sma_5'] = volume_sma_5
        features['volume_sma_20'] = volume_sma_20
        features['volume_ratio'] = self.safe_divide(data['Volume'], volume_sma_20, 1.0)
        
        # Price position in range - güvenli bölme
        high_20 = data['High'].rolling(20).max()
        low_20 = data['Low'].rolling(20).min()
        range_20 = high_20 - low_20
        features['price_position'] = self.safe_divide(
            (data['Close'] - low_20), range_20, 0.5
        )
        
        # Feature matrisini temizle
        features = self.clean_features(features)
        
        return features
    
    def create_target_variable(self, data: pd.DataFrame, prediction_horizon: int = 1) -> pd.Series:
        """Hedef değişken oluşturur (gelecekteki fiyat değişimi)"""
        future_return = data['Close'].shift(-prediction_horizon) / data['Close'] - 1
        return future_return.fillna(0)
    
    def predict_price(self, data: pd.DataFrame, technical_indicators: Dict, 
                     prediction_horizon: int = 1) -> Dict:
        """Gelecek fiyat tahminleri yapar"""
        
        if not self.is_trained:
            return {"error": "Modeller henüz eğitilmemiş. Önce train_models() çağırın."}
        
        try:
            # Son veri noktası için özellikler hazırla
            features = self.prepare_features(data, technical_indicators)
            last_features = features.iloc[-1:][self.feature_names]
            
            # NaN kontrolü ve temizleme
            if last_features.isna().any().any():
                last_features = last_features.fillna(method='ffill').fillna(0)
            
            # Infinity kontrolü
            if np.isinf(last_features.values).any():
                last_features = last_features.replace([np.inf, -np.inf], 0)
            
            current_price = data['Close'].iloc[-1]
            predictions = {}
            valid_predictions = []
            
            for model_name, model in self.trained_models.items():
                try:
                    if model_name == 'svr':
                        # SVR için ölçeklendirilmiş veri kullan
                        last_features_scaled = self.scaler.transform(last_features)
                        pred_scaled = model.predict(last_features_scaled)[0]
                        predicted_return = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]
                    else:
                        predicted_return = model.predict(last_features)[0]
                    
                    # NaN ve infinity kontrolü
                    if np.isnan(predicted_return) or np.isinf(predicted_return):
                        print(f"⚠️ {model_name} modeli NaN/inf döndürdü, atlanıyor")
                        continue
                    
                    # Zaman dilimine göre gerçekçi sınırlar
                    if prediction_horizon <= 1:
                        # 1 günlük tahmin: Çok sınırlı hareket
                        max_daily_change = 0.10  # %10
                        predicted_return = np.clip(predicted_return, -max_daily_change, max_daily_change)
                    elif prediction_horizon <= 7:
                        # 1 haftalık tahmin: Orta düzey hareket
                        max_weekly_change = 0.25  # %25
                        predicted_return = np.clip(predicted_return, -max_weekly_change, max_weekly_change)
                    elif prediction_horizon <= 30:
                        # 1 aylık tahmin: Daha geniş hareket
                        max_monthly_change = 0.50  # %50
                        predicted_return = np.clip(predicted_return, -max_monthly_change, max_monthly_change)
                    else:
                        # Çok uzun vadeli: Maksimum sınır
                        predicted_return = np.clip(predicted_return, -0.70, 1.0)
                    
                    predicted_price = current_price * (1 + predicted_return)
                    
                    # Final kontrol
                    if np.isnan(predicted_price) or np.isinf(predicted_price) or predicted_price <= 0:
                        print(f"⚠️ {model_name} modeli geçersiz fiyat döndürdü, atlanıyor")
                        continue
                    
                    # Ek güvenlik: Fiyat değişim oranını tekrar kontrol et
                    price_change_ratio = abs((predicted_price - current_price) / current_price)
                    max_allowed_change = {
                        1: 0.10,    # 1 gün: %10
                        7: 0.25,    # 1 hafta: %25  
                        30: 0.50    # 1 ay: %50
                    }.get(prediction_horizon, 0.10)
                    
                    if price_change_ratio > max_allowed_change:
                        # Aşırı değişimi sınırla
                        if predicted_price > current_price:
                            predicted_price = current_price * (1 + max_allowed_change)
                        else:
                            predicted_price = current_price * (1 - max_allowed_change)
                        print(f"⚠️ {model_name} modeli aşırı tahmin düzeltildi: {price_change_ratio:.2%} -> {max_allowed_change:.0%}")
                    
                    predictions[model_name] = predicted_price
                    valid_predictions.append(predicted_price)
                    
                except Exception as e:
                    print(f"⚠️ Model {model_name} tahmininde hata: {str(e)}")
                    continue
            
            # Ensemble tahmini hesapla
            if valid_predictions:
                ensemble_prediction = np.mean(valid_predictions)
                predictions['ensemble'] = ensemble_prediction
                
                # Basit format döndür - app.py ile uyumlu
                return predictions
            else:
                return {"error": "Hiçbir model geçerli tahmin üretemedi"}
                
        except Exception as e:
            return {"error": f"Tahmin hesaplama hatası: {str(e)}"}
    
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