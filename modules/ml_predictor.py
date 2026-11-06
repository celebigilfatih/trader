import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from numbers import Integral, Real
from sklearn.utils.validation import check_is_fitted
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
        self.clip_bounds = {}
        self.prediction_horizon = None
    
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
        
    @staticmethod
    def _validate_horizon(horizon):
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, Integral) or horizon <= 0:
            raise ValueError("prediction_horizon must be a positive integer")
        return int(horizon)

    @staticmethod
    def _validate_data(data):
        if not data.index.is_unique or not data.index.is_monotonic_increasing:
            raise ValueError("OHLCV index must be unique and increasing")
        for name in ('Open', 'High', 'Low', 'Close', 'Volume'):
            column = data[name]
            if (not pd.api.types.is_numeric_dtype(column.dtype)
                    or pd.api.types.is_bool_dtype(column.dtype)
                    or pd.api.types.is_complex_dtype(column.dtype)):
                raise ValueError(f"{name} must contain real numeric values")

    @staticmethod
    def _fit_clip_bounds(features):
        bounds = {}
        for name in features.select_dtypes(include=[np.number]).columns:
            if 'change' in name or 'return' in name:
                continue
            lower, upper = features[name].quantile([.01, .99])
            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                bounds[name] = [float(lower), float(upper)]
        return bounds

    @staticmethod
    def _apply_clip_bounds(features, bounds):
        result = features.copy()
        for name, (lower, upper) in bounds.items():
            result[name] = result[name].clip(lower=lower, upper=upper)
        return result

    def _reset_training(self):
        self.is_trained = False
        self.trained_models = {}
        self.feature_names = []
        self.clip_bounds = {}
        self.prediction_horizon = None
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

    def train_models(self, data: pd.DataFrame, technical_indicators: Dict,
                     prediction_horizon: int = 1, test_size: float = 0.2) -> Dict:
        """Fit one coherent train-only preprocessing/model package (ADR-0007)."""
        self._reset_training()
        try:
            horizon = self._validate_horizon(prediction_horizon)
            self._validate_data(data)
            features = self.prepare_features(data, technical_indicators)
            target = self.create_target_variable(data, horizon)
            valid = np.isfinite(features.to_numpy()).all(axis=1) & np.isfinite(target.to_numpy())
            positions = np.flatnonzero(valid)
            if len(positions) < 50:
                return {"error": "Yeterli veri yok. En az 50 geçerli veri noktası gerekli.",
                        "available_data": len(positions)}

            train_positions, test_positions = train_test_split(
                positions, test_size=test_size, random_state=42, shuffle=False)
            # Labels are defined in source-row space, even when valid rows have gaps.
            train_positions = train_positions[train_positions + horizon < test_positions[0]]
            if len(train_positions) < 2 or len(test_positions) < 2:
                return {"error": "At least two train/test samples required after horizon purge"}

            x_train, x_test = features.iloc[train_positions], features.iloc[test_positions]
            y_train, y_test = target.iloc[train_positions], target.iloc[test_positions]
            bounds = self._fit_clip_bounds(x_train)
            x_train = self._apply_clip_bounds(x_train, bounds)
            x_test = self._apply_clip_bounds(x_test, bounds)
            scaler, target_scaler = StandardScaler(), MinMaxScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            y_train_scaled = target_scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()

            results, trained = {}, {}
            for name, model in self.models.items():
                try:
                    model.fit(x_train_scaled, y_train_scaled)
                    train_prediction = target_scaler.inverse_transform(
                        np.asarray(model.predict(x_train_scaled)).reshape(-1, 1)).ravel()
                    test_prediction = target_scaler.inverse_transform(
                        np.asarray(model.predict(x_test_scaled)).reshape(-1, 1)).ravel()
                    metrics = {
                        'train_score': float(r2_score(y_train, train_prediction)),
                        'test_score': float(r2_score(y_test, test_prediction)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, test_prediction))),
                        'mae': float(mean_absolute_error(y_test, test_prediction)),
                    }
                    if not all(np.isfinite(value) for value in metrics.values()):
                        raise ValueError("Nonfinite model metrics")
                    results[name] = metrics
                    trained[name] = model
                except Exception as error:
                    results[name] = {'error': str(error)}
            if not trained:
                return {"error": "Hiçbir model başarıyla eğitilemedi", "model_errors": results}

            self.scaler, self.target_scaler = scaler, target_scaler
            self.feature_names = features.columns.tolist()
            self.clip_bounds = bounds
            self.prediction_horizon = horizon
            self.trained_models = trained
            self.is_trained = True
            summary = {key: float(np.mean([results[name][key] for name in trained]))
                       for key in ('train_score', 'test_score', 'rmse', 'mae')}
            return {**summary, 'models': results, 'successful_models': list(trained),
                    'feature_count': len(self.feature_names),
                    'train_samples': len(train_positions), 'test_samples': len(test_positions)}
        except Exception as error:
            return {"error": f"Model eğitim hatası: {error}"}

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
        horizon = self._validate_horizon(prediction_horizon)
        close = data['Close']
        if close.empty:
            return pd.Series(index=close.index, name=close.name, dtype=float)
        if (not pd.api.types.is_numeric_dtype(close.dtype)
                or pd.api.types.is_bool_dtype(close.dtype)
                or pd.api.types.is_complex_dtype(close.dtype)):
            raise ValueError("Close must contain real numeric values")
        close = close.astype(float)
        if horizon >= len(close):
            return pd.Series(np.nan, index=close.index, name=close.name, dtype=float)
        future = close.shift(-horizon)
        valid = np.isfinite(close) & (close > 0) & np.isfinite(future) & (future > 0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            target = future / close - 1
        return target.where(valid & np.isfinite(target))

    def predict_price(self, data: pd.DataFrame, technical_indicators: Dict, 
                     prediction_horizon: int = 1) -> Dict:
        """Gelecek fiyat tahminleri yapar"""
        
        if not self.is_trained:
            return {"error": "Modeller henüz eğitilmemiş. Önce train_models() çağırın."}
        
        try:
            horizon = self._validate_horizon(prediction_horizon)
            self._validate_data(data)
            if horizon != self.prediction_horizon:
                raise ValueError("Prediction horizon differs from training horizon")
            features = self.prepare_features(data, technical_indicators)
            last_features = features.iloc[-1:][self.feature_names]
            last_features = self._apply_clip_bounds(last_features, self.clip_bounds)
            last_features_scaled = self.scaler.transform(last_features)
            current_price = data['Close'].iloc[-1]
            if not np.isfinite(current_price) or current_price <= 0:
                raise ValueError("Current price must be positive and finite")
            predictions = {}
            valid_predictions = []
            
            for model_name, model in self.trained_models.items():
                try:
                    pred_scaled = model.predict(last_features_scaled)[0]
                    predicted_return = self.target_scaler.inverse_transform([[pred_scaled]])[0][0]

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
    
    @staticmethod
    def _validate_model_package(package):
        """Validate a trusted deserialized package without touching live state."""
        try:
            if not isinstance(package, dict) or type(package.get('format_version')) is not int or package['format_version'] != 1:
                raise ValueError("Unsupported model format; retrain legacy models")
            if package['is_trained'] is not True:
                raise ValueError("Model package is not trained")
            MLPredictor._validate_horizon(package['prediction_horizon'])
            names = package['feature_names']
            if (not isinstance(names, list) or not names
                    or not all(isinstance(name, str) and name for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError("Invalid feature schema")
            count = len(names)
            preprocessing = package['preprocessing']
            if not isinstance(preprocessing, dict) or not isinstance(preprocessing.get('clip_bounds'), dict):
                raise ValueError("Missing fitted clipping bounds")
            for name, pair in preprocessing['clip_bounds'].items():
                if (name not in names or 'change' in name or 'return' in name
                        or not isinstance(pair, (list, tuple)) or len(pair) != 2
                        or not all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                                   and np.isfinite(value) for value in pair)
                        or pair[0] >= pair[1]):
                    raise ValueError("Invalid clipping bounds")

            scaler, target = package['scaler'], package['target_scaler']
            if not isinstance(scaler, StandardScaler) or not isinstance(target, MinMaxScaler):
                raise ValueError("Invalid scaler type")
            if not scaler.with_mean or not scaler.with_std or target.feature_range != (0, 1):
                raise ValueError("Invalid scaler configuration")
            for obj, size, attributes in (
                    (scaler, count, ('mean_', 'var_', 'scale_')),
                    (target, 1, ('scale_', 'min_', 'data_min_', 'data_max_', 'data_range_'))):
                check_is_fitted(obj)
                samples = obj.n_samples_seen_
                if (obj.n_features_in_ != size or isinstance(samples, (bool, np.bool_))
                        or not isinstance(samples, Real) or not np.isfinite(samples)
                        or samples < 2 or samples != int(samples)):
                    raise ValueError("Invalid fitted scaler dimensions")
                for attribute in attributes:
                    values = np.asarray(getattr(obj, attribute))
                    if values.shape != (size,) or not np.isfinite(values).all():
                        raise ValueError("Invalid fitted scaler parameters")
                if np.any(np.asarray(obj.scale_) <= 0):
                    raise ValueError("Invalid scaler scale")
            if (np.any(np.asarray(scaler.var_) < 0) or np.any(np.asarray(target.data_range_) < 0)
                    or np.any(target.data_max_ < target.data_min_)
                    or not np.allclose(target.data_range_, target.data_max_ - target.data_min_)):
                raise ValueError("Invalid scaler ranges")
            if scaler.n_samples_seen_ != target.n_samples_seen_:
                raise ValueError("Scaler training sample counts mismatch")
            # sklearn's near-constant feature rule includes floating-point error.
            var, mean = np.asarray(scaler.var_), np.asarray(scaler.mean_)
            eps = np.finfo(np.float64).eps
            count_seen = scaler.n_samples_seen_
            constant = var <= count_seen * eps * var + (count_seen * mean * eps) ** 2
            expected_scale = np.where(constant, 1., np.sqrt(var))
            if not np.allclose(scaler.scale_, expected_scale, rtol=1e-12, atol=1e-15):
                raise ValueError("Inconsistent standard scaler scale")
            data_range = np.asarray(target.data_range_)
            denominator = np.where(data_range < 10 * np.finfo(data_range.dtype).eps, 1., data_range)
            if (not np.allclose(target.scale_, 1. / denominator, rtol=1e-12, atol=1e-15)
                    or not np.allclose(target.min_, -target.data_min_ * target.scale_,
                                       rtol=1e-12, atol=1e-15)):
                raise ValueError("Inconsistent target scaler scale or offset")
            if not np.array_equal(getattr(scaler, 'feature_names_in_', None), names):
                raise ValueError("Scaler feature order mismatch")
            models = package['models']
            if not isinstance(models, dict) or not models:
                raise ValueError("Missing trained models")
            for name, model in models.items():
                if not isinstance(name, str) or not name or not callable(getattr(model, 'predict', None)):
                    raise ValueError("Invalid model entry")
                check_is_fitted(model)
                if model.n_features_in_ != count:
                    raise ValueError("Model feature dimensions mismatch")
        except (KeyError, AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid model package: {error}") from error

    def save_models(self, filepath: str) -> None:
        """Save a complete format1 package; only trusted local joblib is supported."""
        model_data = {
            'format_version': 1,
            'models': self.trained_models,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'preprocessing': {'clip_bounds': self.clip_bounds},
            'prediction_horizon': self.prediction_horizon,
        }
        self._validate_model_package(model_data)
        joblib.dump(model_data, filepath)

    def load_models(self, filepath: str) -> None:
        """Load trusted local joblib. Format checks do not secure deserialization."""
        model_data = joblib.load(filepath)
        self._validate_model_package(model_data)
        # Publish only after every field/model/scaler has passed validation.
        self.trained_models = model_data['models']
        self.scaler = model_data['scaler']
        self.target_scaler = model_data['target_scaler']
        self.feature_names = model_data['feature_names']
        self.clip_bounds = model_data['preprocessing']['clip_bounds']
        self.prediction_horizon = int(model_data['prediction_horizon'])
        self.is_trained = True
