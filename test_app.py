#!/usr/bin/env python3
"""
BIST Teknik Analiz Uygulaması Test Script'i
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Modules klasörünü sys.path'a ekle
sys.path.append('modules')

def test_modules():
    """Tüm modüllerin doğru çalışıp çalışmadığını test eder"""
    
    print("🧪 Modül testleri başlatılıyor...")
    print("=" * 50)
    
    # Test 1: Config modülü
    try:
        from modules.config import BIST_SYMBOLS, INDICATORS_CONFIG, ALERT_CONFIG
        print("✅ Config modülü: OK")
        print(f"   - {len(BIST_SYMBOLS)} hisse kodu yüklendi")
        print(f"   - {len(INDICATORS_CONFIG)} indikatör konfigürasyonu yüklendi")
    except Exception as e:
        print(f"❌ Config modülü: {e}")
        return False
    
    # Test 2: Data Fetcher modülü
    try:
        from modules.data_fetcher import BISTDataFetcher
        fetcher = BISTDataFetcher()
        print("✅ Data Fetcher modülü: OK")
        
        # Test veri çekme
        print("   📊 Test veri çekiliyor (THYAO)...")
        df = fetcher.get_stock_data("THYAO.IS", period="1mo", interval="1d")
        if df is not None and not df.empty:
            print(f"   - {len(df)} günlük veri çekildi")
        else:
            print("   ⚠️ Veri çekilemedi (internet bağlantısı gerekli)")
            
    except Exception as e:
        print(f"❌ Data Fetcher modülü: {e}")
        return False
    
    # Test 3: Technical Analysis modülü
    try:
        from modules.technical_analysis import TechnicalAnalyzer
        
        # Test verileri oluştur
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        test_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, len(dates)),
            'High': np.random.uniform(105, 115, len(dates)),
            'Low': np.random.uniform(95, 105, len(dates)),
            'Close': np.random.uniform(100, 110, len(dates)),
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        analyzer = TechnicalAnalyzer(test_data)
        analyzer.add_indicator('ema_21')
        analyzer.add_indicator('ema_50')
        analyzer.add_indicator('rsi')
        analyzer.add_indicator('macd')
        analyzer.add_indicator('supertrend')
        analyzer.add_indicator('ott')
        
        print("✅ Technical Analysis modülü: OK")
        latest_indicators = analyzer.get_latest_indicators()
        print(f"   - {len(latest_indicators)} indikatör hesaplandı")
        
    except Exception as e:
        print(f"❌ Technical Analysis modülü: {e}")
        return False
    
    # Test 4: Alert System modülü
    try:
        from modules.alert_system import AlertSystem
        alert_system = AlertSystem()
        
        # Test sinyal üretimi
        if 'analyzer' in locals():
            signal = alert_system.generate_signal(analyzer)
            print("✅ Alert System modülü: OK")
            print(f"   - Test sinyali: {signal}")
        
    except Exception as e:
        print(f"❌ Alert System modülü: {e}")
        return False
    
    print("=" * 50)
    print("🎉 Tüm testler başarıyla tamamlandı!")
    return True

def test_dependencies():
    """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder"""
    
    print("📦 Bağımlılık testleri...")
    print("=" * 30)
    
    dependencies = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'yfinance',
        'ta',
        'requests',
        'beautifulsoup4'
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} (eksik)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Eksik bağımlılıklar: {', '.join(missing_deps)}")
        print("📦 Kurmak için: pip install -r requirements.txt")
        return False
    
    print("✅ Tüm bağımlılıklar yüklü!")
    return True

def main():
    """Ana test fonksiyonu"""
    print("🚀 BIST Teknik Analiz Uygulaması - Test Suite")
    print("=" * 60)
    
    # Bağımlılık testleri
    if not test_dependencies():
        sys.exit(1)
    
    print("\n")
    
    # Modül testleri
    if not test_modules():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎯 Test Sonucu: BAŞARILI")
    print("✨ Uygulama kullanıma hazır!")
    print("🚀 Başlatmak için: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 