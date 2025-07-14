#!/usr/bin/env python3
"""
BIST Teknik Analiz Uygulaması Çalıştırma Script'i
"""

import os
import sys
import subprocess
import time

def check_requirements():
    """Gerekli paketlerin yüklü olup olmadığını kontrol eder"""
    try:
        import streamlit
        import pandas
        import plotly
        import yfinance
        import ta
        print("✅ Tüm gerekli paketler yüklü!")
        return True
    except ImportError as e:
        print(f"❌ Eksik paket: {e}")
        print("📦 Gerekli paketleri yüklemek için: pip install -r requirements.txt")
        return False

def main():
    """Ana fonksiyon"""
    print("🚀 BIST Teknik Analiz Uygulaması başlatılıyor...")
    print("=" * 50)
    
    # Gereksinim kontrolü
    if not check_requirements():
        sys.exit(1)
    
    # Modülleri kontrol et
    if not os.path.exists("modules"):
        print("❌ Modüller klasörü bulunamadı!")
        sys.exit(1)
    
    # Streamlit uygulamasını çalıştır
    try:
        print("🌐 Web uygulaması başlatılıyor...")
        print("📱 Tarayıcınızda http://localhost:8501 adresini açın")
        print("🛑 Uygulamayı durdurmak için Ctrl+C tuşlarına basın")
        print("=" * 50)
        
        # Streamlit komutunu çalıştır
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Uygulama durduruldu!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Uygulama çalıştırılırken hata oluştu: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 