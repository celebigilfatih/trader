# 📈 BIST Teknik Analiz Uygulaması

Borsa İstanbul (BIST) hisselerini analiz eden, teknik indikatörler hesaplayan ve al-sat sinyalleri veren modern bir finansal analiz uygulaması.

> **Olgunluk notu:** Repository mevcut haliyle yerel çalışan bir prototiptir.
> Çıktılar yatırım tavsiyesi değildir; piyasa verileri ve haber akışı üçüncü taraf
> servislerin erişilebilirliğine bağlıdır.

## CDSK proje yönetimi

Bu repository **Celebigil Development Standard Kit (CDSK) 0.1.0** kullanır.
Geliştirmeye başlamadan önce `AGENTS.md`,
`docs/00-product/CONSTITUTION.md`, `PROJECT_BOOT.md` ve görevle ilgili belgeler
okunmalıdır. Güncel durum, açık riskler ve önerilen sonraki adım
`PROJECT_BOOT.md` içindedir.

### Hedeflenen ürün kurgusu

- Kapalı beta, paper/simülasyon odaklı, günlük karar cockpit’i.
- Mimari hedef: FastAPI + Next.js + PostgreSQL + Hostinger/Coolify.
- Güvenlik hedefi: davet + passwordless e-posta link.
- AI çıktıları: core karar akışını etkilemeyen deneysel modül.

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
Node.js 18+ ve npm (Next.js arayüzü için)
```

### 2. Depoyu Klonlayın
```bash
git clone https://github.com/your-username/bist-teknik-analiz.git
cd bist-teknik-analiz
```

### 3. Bağımlılıkları Kurun
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 4. Geliştirme ortamını tek komutla başlatın
```bash
python3 run_dev.py
```

Bu komut backend (`127.0.0.1:8000`) ile Next.js cockpit’i (`127.0.0.1:3000`) birlikte
başlatır.

Opsiyonel parametre örnekleri:
```bash
python3 run_dev.py --api-host 127.0.0.1 --api-port 8000 --front-host 127.0.0.1 --front-port 3000
python3 run_dev.py --install-frontend-deps
python3 run_dev.py --no-npm-install-check
python3 run_dev.py --ready-timeout 45
python3 run_dev.py --check-interval 0.5 --shutdown-timeout 8
```

`--ready-timeout` backend/front-end hazır olma bekleme süresini (saniye),
`--check-interval` health kontrol aralığını (saniye),
`--shutdown-timeout` kapanışta `SIGTERM` sonrası bekleme süresini belirler.

Tarayıcınızdan `http://localhost:3000` adresini açın. Uygulama API ile çalışır; önce
`/api/v1/health` üzerinden backend’in açık olduğundan emin olun.
Cockpit, demo auth akışıyla token üretip oturum açma, teknik analiz, RSI tarama,
watchlist ve paper portföy görüntüsü gibi akışları içerir.

Windows kullanıcıları için:
```bash
run_dev.bat
```

### 5. API'yı ayrı başlatın (gelişmiş kullanım)
```bash
python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Alternatif:
```bash
python3 run_api.py --reload
```

### 6. Next.js Cockpit'u ayrı başlatın
Farklı bir terminalde:

```bash
cd frontend
npm install
npm run dev
```

### 7. Eski Streamlit arayüzünü çalıştırın (legacy)
`run.py` ile mevcut klasik arayüz için:
```bash
python3 run.py
```

Tarayıcınızda `http://localhost:8501` adresini açın.

### 8. Backend smoke test (pytest)
```bash
python3 -m pytest backend/tests
```

Eğer `pytest` kurulu değilse:
```bash
python3 -m unittest discover -s backend/tests -p 'test_*.py'
```

### 9. Next.js derleme kontrolü
```bash
cd frontend
npm run build
```

### 10. `run_dev` geliştirme orkestrasyonu smoke testi
```bash
python3 scripts/run_dev_smoke.py
```

Kısa süreli bu test, `run_dev.py` ile backend ve frontend’i birlikte başlatır,
hazır olma kontrollerini doğrular; backend sağlığı + auth invite/magic + watchlist
ve portfolio kontrat doğrulamaları ile frontend ana ekran marker doğrulamasını
yaparak ardından scripti güvenli şekilde durdurur.
Özel port/timeout değerleri için:

```bash
python3 scripts/run_dev_smoke.py --api-port 8020 --front-port 3020 --duration 3 --ready-timeout 20
```

### 10.1. Playwright ile opsiyonel UI smoke test

```bash
make smoke-ui
```

Bu senaryo backend’den oturum alıp, frontend’de temel watchlist/portföy eylemlerini
tarayıcıyla doğrular.

### 10.2. Stack ile tek komutta UI smoke testi

```bash
python3 scripts/run_frontend_ui_smoke_with_stack.py
```

Bu akış, `run_dev.py` ile backend ve frontend’i birlikte başlatır; hazır olma
kontrollerini tamamladıktan sonra `scripts/run_frontend_ui_smoke.mjs` çalıştırılır.
Test bitince stack güvenli şekilde kapatılır.

Aynı işlem için Makefile hedefi de kullanılabilir:

```bash
make smoke-ui-stack
```

Playwright yüklü değilse `scripts/run_frontend_ui_smoke.mjs` size kurulum yönergesi
gösterir. Kurulum:

```bash
cd frontend
npm i -D playwright
npm exec playwright install
```

### 11. Hızlı yerel görevler (`Makefile`)

```bash
make run-dev         # run_dev ile ortamı ayağa kaldır
make test            # backend testlerini çalıştır
make build-frontend  # frontend build
make smoke           # kısa run_dev smoke check'i
make smoke-ui        # isteğe bağlı tarayıcı tabanlı UI smoke
make smoke-ui-stack  # stack başlatıp UI smoke çalıştırır
make smoke-all       # smoke + smoke-ui-stack ardışık koşar
make ci-e2e          # make ci-core + make smoke-ui-stack
make ci-full         # make ci-e2e (tam akış: test + build + e2e)
make qa              # günlük kapanış QA: tarayıcı e2e dahil hızlı kapalı-çevrim akış
make ci-core         # make ci'ye temel sağlayan deps+test+build adımı (smoke olmadan)
make ci              # test + frontend build + smoke
make clean           # geçici dosyaları temizle
```

### 12. CI akışı (GitHub Actions)

Repo için otomatik CI, `.github/workflows/ci.yml` ile tanımlıdır ve:
- `make ci` hedefini çalıştırır (bağımlılık yükleme, testler, frontend build, smoke)
- `make ci-e2e` hedefini çalıştırır (`make ci-core` + `make smoke-ui-stack`)
- `make ci-full` hedefini çalıştırır (`make ci-e2e`)
- Pull request ve `main` branch’e push’larda otomatik olarak tetiklenir.

### 12.1. Günlük doğrulama playbook'u (öneri)

- Günlük geliştirme sonrası (hızlı): `make smoke`
- Fonksiyonel doğrulama: `make smoke-ui-stack` (opsiyonel, Playwright gerektirir)
- CI’ye yakın doğrulama: `make ci-e2e` (`make ci-core` + `make smoke-ui-stack`)
- Kapanış QA: `make qa`
- Derin kapanış doğrulaması: `make ci-full`

`make ci-full`, `smoke-ui-stack` çağrısında ikinci UI stack için farklı portlar kullandığı için
ardışık çalıştırmalarda port çakışmalarını azaltır. CI pipeline’da şu an sadece
`make ci` çağrısı çalışır; `make ci-full` yerel kapama/ön kontrol akışı olarak
kullanılabilir.

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
