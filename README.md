# 📈 BIST Teknik Analiz Uygulaması

Borsa İstanbul (BIST) hisselerini analiz eden, teknik indikatörler hesaplayan ve al-sat sinyalleri veren modern bir finansal analiz uygulaması.

## 🚀 Özellikler

### 📊 Teknik Analiz
- **Hareketli Ortalamalar**: SMA 20, SMA 50, EMA 12, EMA 26
- **Momentum İndikatörleri**: RSI, Stokastik, Williams %R
- **Trend İndikatörleri**: MACD, Bollinger Bantları, CCI
- **Destek-Direnç**: Otomatik seviye tespiti
- **Grafik Desenleri**: Double Top/Bottom, Üçgen, Bayrak

### 🚨 Alert Sistemi
- **Akıllı Al-Sat Sinyalleri**: Birden fazla indikatörü birleştiren algoritmik sinyal üretimi
- **Fiyat Alertleri**: Hedef fiyat ve stop-loss uyarıları
- **Teknik Alertler**: RSI aşırı alım/satım, hacim patlaması, büyük fiyat hareketleri
- **Çoklu Alert Yöntemleri**: Email, Telegram, Desktop bildirimleri

### 📈 Görselleştirme
- **Interactive Grafikler**: Plotly ile profesyonel grafikler
- **Candlestick Grafikler**: OHLC verileri ile mum grafikler
- **Çoklu Zaman Dilimleri**: 1 ay'dan 2 yıl'a kadar
- **Gerçek Zamanlı Veriler**: Yahoo Finance API ile güncel veriler

### 💼 Desteklenen Hisseler
30+ popüler BIST hissesi:
- **Bankacılık**: AKBNK, GARAN, ISCTR, HALKB, VAKBN
- **Sanayi**: THYAO, TUPRS, ARCLK, EREGL, ASELS
- **Perakende**: BIMAS, MGROS, SOKM
- **Holding**: KCHOL, SAHOL, DOHOL
- **Teknoloji**: TCELL, PGSUS
- **Ve daha fazlası...**

## 🛠️ Kurulum

### 1. Gereksinimler
```bash
Python 3.8+
pip (Python package manager)
```

### 2. Depoyu Klonlayın
```bash
git clone https://github.com/your-username/bist-teknik-analiz.git
cd bist-teknik-analiz
```

### 3. Bağımlılıkları Kurun
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı Çalıştırın
```bash
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresini açın.

## 📖 Kullanım Kılavuzu

### Temel Kullanım
1. **Hisse Seçimi**: Sol menüden istediğiniz hisseyi seçin
2. **Zaman Aralığı**: Analiz edilecek süreyi belirleyin
3. **İndikatörler**: Görmek istediğiniz teknik indikatörleri seçin
4. **Analiz**: Grafik ve sinyaller otomatik olarak güncellenir

### Alert Kurulumu
1. **Alertleri Aktifleştir**: Sol menüden alert seçeneğini açın
2. **Yöntem Seçimi**: Email, Telegram veya Desktop bildirimleri
3. **Otomatik Uyarılar**: Sinyal oluştuğunda bildirim alın

### Sinyal Yorumlama
- **🟢 AL**: Birden fazla indikatör alım sinyali veriyor
- **🔴 SAT**: Birden fazla indikatör satım sinyali veriyor  
- **🟡 BEKLE**: Karışık sinyaller, pozisyon almayın

## 🔧 Gelişmiş Özellikler

### Özel İndikatör Ekleme
```python
# modules/technical_analysis.py dosyasını düzenleyin
def _calculate_custom_indicator(self, indicator_name: str) -> None:
    # Özel indikatör hesaplama logic'i
    pass
```

### Alert Konfigürasyonu
```python
# modules/config.py dosyasında alert ayarlarını değiştirin
ALERT_CONFIG = {
    "rsi_oversold": 30,        # RSI aşırı satım seviyesi
    "rsi_overbought": 70,      # RSI aşırı alım seviyesi
    "volume_spike_multiplier": 2.0,  # Hacim patlaması çarpanı
    "price_change_threshold": 5.0,   # Fiyat değişim eşiği (%)
}
```

### Email Alert Kurulumu
```python
# Email ayarlarını yapılandırın
smtp_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'password': 'your-app-password'
}
```

## 📊 Teknik İndikatörler Açıklaması

### Hareketli Ortalamalar
- **SMA 20**: Son 20 günün basit ortalaması, kısa vadeli trend
- **SMA 50**: Son 50 günün basit ortalaması, orta vadeli trend
- **Golden Cross**: SMA 20'nin SMA 50'yi yukarı kesmesi (AL sinyali)
- **Death Cross**: SMA 20'nin SMA 50'yi aşağı kesmesi (SAT sinyali)

### RSI (Relative Strength Index)
- **0-30**: Aşırı satım bölgesi (AL fırsatı)
- **30-70**: Normal bölge
- **70-100**: Aşırı alım bölgesi (SAT fırsatı)

### MACD
- **MACD > Sinyal**: Yukarı trend
- **MACD < Sinyal**: Aşağı trend
- **Sıfır çizgisi kesişimi**: Trend değişimi

### Bollinger Bantları
- **Fiyat alt banda değerse**: Potansiyel AL sinyali
- **Fiyat üst banda değerse**: Potansiyel SAT sinyali
- **Bantların daralması**: Volatilite artışı beklentisi

## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Bir Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasını inceleyin.

## ⚠️ Uyarılar

- **Yatırım Danışmanlığı Değildir**: Bu uygulama sadece analiz amaçlıdır
- **Risk Uyarısı**: Finansal piyasalarda kayıp riskiniz bulunmaktadır
- **Kendi Kararınızı Verin**: Yatırım kararlarınızı kendi sorumluluğunuzda alın
- **Backtesting Yapın**: Stratejilerinizi geçmiş verilerle test edin

## 📞 İletişim

- **Email**: your-email@domain.com
- **GitHub**: [GitHub Profile](https://github.com/your-username)
- **LinkedIn**: [LinkedIn Profile](https://linkedin.com/in/your-profile)

## 🏆 Teşekkürler

- **Yahoo Finance**: Finansal veriler için
- **Streamlit**: Web uygulaması framework'ü için
- **Plotly**: İnteraktif grafikler için
- **TA-Lib**: Teknik analiz hesaplamaları için

---

**🚀 Başarılı yatırımlar dileriz!** 