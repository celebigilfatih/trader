@echo off
echo 🚀 BIST Teknik Analiz Uygulaması
echo ================================
echo.
echo 📦 Gerekli paketler kontrol ediliyor...
pip install -r requirements.txt
echo.
echo 🌐 Uygulama başlatılıyor...
echo 📱 Tarayıcınızda http://localhost:8501 adresini açın
echo 🛑 Durdurmak için Ctrl+C tuşlarına basın
echo.
streamlit run app.py
echo.
echo 👋 Uygulama durduruldu!
pause 