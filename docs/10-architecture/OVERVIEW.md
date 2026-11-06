# Architecture Overview

## System Context

Mevcut yerel geliştirme yapısı `backend/app` FastAPI servisi ile `frontend/app`
Next.js cockpit'ini `run_dev.py` üzerinden birlikte çalıştırır. Teknik hesaplama
çekirdeği `modules/` altındadır. Streamlit uygulaması legacy olarak korunur;
haber, ML, grafik ve PDF gibi miras akışların tamamı yeni cockpit'e taşınmış değildir.

## Layers and Responsibilities

1. **Presentation:** `frontend/app` cockpit; `app.py` legacy Streamlit sayfaları.
2. **Market data:** `modules/data_fetcher.py`.
3. **Analysis:** `technical_analysis.py`, patern modülleri ve
   `alert_system.py`.
4. **Research automation:** `stock_screener.py`, `day_trader.py`.
5. **Experimental intelligence:** `simple_ml_predictor.py`, `ml_predictor.py`,
   `sentiment_analyzer.py`.
6. **Configuration:** `modules/config.py`.
7. **API/services:** `backend/app/routers.py`, provider ve bellek içi auth,
   watchlist, paper portföy servisleri. PostgreSQL henüz uygulanmış değildir.

`app.py` yaklaşık 6.500 satırdır ve presentation ile orchestration
sorumluluklarını birlikte taşır; bu mevcut durumdur, hedef mimari kararı değildir.

## Target State (Accepted, incremental migration)

- **Strangler migration:** Mevcut Streamlit arayüzü korunur; yeni karar-katmanı
  ve analitik API'si `FastAPI` ile ayrı bir süreçte inşa edilir.
- **Frontend:** Kısa vadede `Next.js` cockpit UI, uzun vadede Streamlit sadece
  yedek/arama ekranı olarak kalır.
- **Backend:** `FastAPI` içinde `/api/v1` endpointleri, PostgreSQL merkezli
  depolama, jobs ve domain servisleri.
- **Background work:** Coolify cron/job üzerinden günlük ingest, tarama ve
  analizleri gün sonunda tetiklenir.
- **Auth ve yetki:** Kapalı beta için davet + passwordless e-posta link ile oturum.
- **Ops:** Coolify rollback, health check ve günlük yedek alma ile birlikte
  izlenebilir bir dağıtım hattı hedeflenir.

## Current-state / Target-state farkı

- Mevcut durum: yerel FastAPI + Next.js prototipi ve paralel korunan legacy Streamlit.
- Hedef durum: API-first, test edilebilir domain çekirdeği + yeni web kabuğu.
- Geçiş yöntemi: kritik akışlar tek tek taşınır; mevcut kullanıcı deneyimi korunur.

## Main Integrations

- Yahoo Finance (`yfinance`): fiyat, endeks ve şirket bilgisi.
- Beş harici RSS kaynağı: finans haberleri.
- SMTP: yalnızca `AlertSystem.send_email_alert` çağrılırsa e-posta gönderimi.

## AI Components

Generative AI veya uzak model servisi yoktur. “AI Tahminleri” yerel
scikit-learn modellerini; sentiment ise VADER/TextBlob hesaplamalarını kullanır.
Bazı sentiment fonksiyonları açıkça mock/demo veri üretir.

## Data and Storage

- Piyasa/haber verisi çalışma anında bellekte pandas nesneleri ve Streamlit
  session state/cache içinde tutulur.
- `portfolio_data.json` repository'de izlenen yerel bir finansal veri örneğidir;
  güncel UI akışında kullanımına rastlanmadı.
- `save_alert_history` çağrılırsa çalışma dizinine CSV yazar.
- Eğitilmiş ML modeli `joblib` ile isteğe bağlı kaydedilebilir.
- API auth, watchlist ve paper portföy kayıtları süreç belleğindedir;
  backend yeniden başlatılınca kaybolur. Bu yapı üretim kalıcılığı değildir.

## Deployment Model

Yerel API + cockpit için `python3 run_dev.py`, legacy için `python3 run.py`
kullanılır. `.github/workflows/ci.yml` içinde `make ci` tanımlıdır;
bu workflow üretime dağıtım yapmaz. Coolify/PostgreSQL/worker, backup ve rollback
hedefleri henüz tamamlanmış üretim kabiliyeti olarak değerlendirilmez.

## Key Risks and ADR Links

- Finansal karar desteğinde veri tazeliği ve sahte/gerçek ayrımı kritik.
- API provider'ında sınırlı retry/backoff ve API health endpoint'i var;
  `timeout_seconds` artık gerçek history çağrısına iletiliyor; toplam işlem
  deadline'ı değildir. Accepted ADR-0005 RSS connect/read timeout ve sınırlı retry
  sağlar; RSS/Yahoo kaynak tanıları bellektedir. Merkezi üretim gözlemi açık.
- UI ve domain hesaplamalarının büyük ölçüde tek dosyada birleşmesi değişiklik
  riskini yükseltiyor.
- 412 backend testi dış ağ bağlantıları engellenerek geçiyor (2026-11-06);
  Windows asyncio için yerel loopback serbest. Legacy `test_app.py` ve
  tarayıcı smoke akışı dış sağlayıcılara bağlı kalabilir.
- Stochastic/Williams %R/CCI çekirdeği 17 gerçek hesaplama testiyle sınandı.
  Sonraki Accepted ADR-0009, API `stoch` seçicisini mevcut `stoch_k` serisine
  eşleyerek OS-01'i kapattı.7 gerçek HTTP testi K≠D,zero/null,mixed/meta ve400/401
  doğrular; dış ad/skaler biçim ve core/legacy seriler korunur.
  [Uygulama kanıtı](../20-execution/2026-11-06-adr0009-implementation.md).
- Önceki VWEMA/Supertrend13 testi geçerli sabit-seed/yeterli veri alt kümesini,
  gerçek API ve sinyal okuyucularını kapsadı. O aşamada VST-01 VWEMA seed lookahead,
  VST-02 başlangıç sonrası pozitif hacmin sadeleşmesi ve VST-03 kısa Supertrend
  fiyat/yön/seed tutarsızlığı açıktı. Mevcut temel bant hesabı canonical
  trailing Supertrend diye doğrulanmadı; [test ve karşı örnek kaydı](../20-execution/2026-11-06-vwema-supertrend-tests.md).
  [ADR-0010](../50-decisions/ADR-0010.md) A paketi **Accepted ve uygulandı**:
  nedensel hacim ağırlığı + sabit Supertrend hazır olma, strict/atomik girdi ve
  nötr eksik kart. VST-01–03 dar dilimi kapandı; canonical bant veya etkinlik iddiası yok.
- Ayrı deterministik cockpit modu 10 senaryodur. 21 kontrollü Streamlit AppTest
  legacy haber/ML fonksiyonları ve gerçek VWEMA kart AST'sini render eder;
  tam uygulama/browser kanıtı değildir.
- BOS mevcut batch hesabında swing için çevredeki gelecek mumları da kullanır;
  tarama sondan 10. mumu içerir, daha yeni 9 mumu dışlar. Eklenen testler bu
  davranışı karakterize eder; canlı/causal strateji veya kârlılık kanıtı değildir.
  BOS güç artışı 100'ü aşabilir; kombinasyon confidence değeri sezgiseldir.
- RSI/MACD sıfır okuma ve güçlü SAT boyut etiketi düzeltildi. DayTrader geçersiz
  ATR/entry/yön girdisini ValueError ile reddeder; hata scan'de sembol bazlı
  yakalanır. Mevcut geçerli hesaplar/kayıt şekli korunur. Accepted ADR-0006 OTT
  seed/yön sözleşmesi uygulanıp gerçek okuyucu zincirinde test edildi. Boş/tek
  gözlem ve geçersiz Close davranışı belirgindir. Genel risk/pozisyon entegrasyonu
  ve üretime hazırlık tamamlanmış değildir; canonical OTT/kârlılık kanıtı yoktur.
- ML için97 hedef test vardır. Accepted ADR-0007 native hedef kuyruğu/purge,
  train-only quantile ve ortak dört-model scaler/inverse, coherent advanced state
  ve format1 joblib sözleşmesini uyguladı. Native dört-model smoke ile ayrı literal
  label boundary hesapları ayrılır; finansal doğruluk/kalibrasyon iddiası yoktur.
  Sonraki Simple state/importance/NaN dar düzeltmeleri mevcut çıktı biçiminde
  tamamlandı. Accepted ADR-0008 tek cap, price/current confidence ve h gözlem
  sunumunu uyguladı. UI yayılım kalibrasyonu/genel model etkinliği ayrı açıktır.
- Kabul edilmiş hedef kararlar: ADR-0001 (kapsam), ADR-0002 (mimari),
  ADR-0003 (kimlik/bildirim). Demo auth, kabul edilmiş üretim davet operasyonunun
  tamamlandığı anlamına gelmez.
