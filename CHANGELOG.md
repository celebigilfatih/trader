# Changelog

## [Unreleased]

### 2026-11-06 — Git checkpoint teslimi ve ADR-0008 Proposed

- Kullanıcı Commit & Push talebiyle70 dosyalık M1 güvenilirlik çalışması `abfc592`
  commit'i olarak `origin/codex/m1-pattern-signal-tests` dalına gönderildi; upstream
  izleme kuruldu. Merge/PR/force-push yok. Yerel otomatik Git kimliği korundu.
- Teslim öncesi331 backend testi, Next.js üretim build/lint/type ve üretilmiş
  frontend üzerinde10/10 deterministik Edge senaryosu tekrar geçti (dış istek0).
- Sonraki adım ADR-0008 Proposed: tek ufuk cap'i, fiyat/current_price confidence
  girdi uyumu ve panelde gözlem ufku/kesin olmayan tarih sunumu. Tasarım bağımsız
  builder incelemesinden geçti; özel uygulama onayı bekliyor. Runtime kodu/test/
  dependency/artifact değişmedi. Backlog/mimari/kalite/handoff güncellendi.

### 2026-11-06 — Simple ML-04–06 dar hata düzeltmeleri

- Her Simple eğitim denemesi eski trained flag/cache'i temizler; nonfinite metric
  success sayılmaz. Inference hatası eski tahmini saklamaz; geçerli retry çalışır.
- Feature importance sabit17 ad yerine RF'nin gerçek fitted kolon metadata'sını
  kullanır; bare10 ve partial optional isim/ağırlık eşleşmesi korunur.
- Nonfinite return/current/derived değer, taşan percent/fiyat ve pozitif olmayan
  fiyat mevcut error sözlüğüne gider; geçerli signal/confidence formülü aynı.
- 13 yeni test önce29 assertion failure, sonra37 hedef green;11 geçerli mutant
  yakalandı, bağımsız inceleme bulgusuz. Toplam331 backend testi/60 Python compile
  ve CDSK yapı/link/ADR/diff kontrolü başarılı; upstream validator ortamda yok.
- Advanced/config/dependency başlangıçla aynı; yeni ADR/şema/model veya commit/
  push yok. Belgeler/handoff güncel; frontend/canlı sağlayıcı tekrar yok. Kalan
  ML-07/08/takvim ve genel M1 açık. Plan `2026-11-06-simple-ml-fixes.md`.

### 2026-11-06 — ADR-0007 onaylı ML sözleşmesi uygulandı

- Kullanıcının somut soruya “Onaylıyorum.” yanıtıyla ADR-0007 Accepted. Bilinmeyen
  hedef NaN, gerçek sıfır korunur; min50 etiket/source-position ufuk purge ve
  train/test en az2, unique/increasing index ve pozitif integer ufuk doğrulanır.
- Global batch quantile kaldırıldı; bounds/scaler yalnız purge edilmiş train'den.
  Dört estimator aynı X scaler/target inverse kullanır. Başarısız advanced retrain
  eski state'i iptal eder; yalnız metric aşamasını da geçen modeller yayımlanır.
- Format1 joblib bounds/ufuk saklar, legacy yeniden eğitim ister. Bozuk/eksik paket
  çalışan instance'ı değiştirmez. Review scaler türetilmiş parametre tutarlılığı
  ve çok büyük ufuk bulguları RED→GREEN kapandı; kullanıcı artifact'ı açılmadı.
- 30 yeni regresyonla70 ML/318 backend testi geçti;15 geçerli mutant yakalandı,
  bağımsız re-review bulgusuz. 59 Python compile ve CDSK yapı/link/ADR/diff
  kontrolleri geçti; upstream validator ortamda yok. Dependency/model config aynı,
  commit/push ve frontend/canlı servis yok; Simple/kalan ML ve genel M1 açık.

### 2026-11-06 — ML-01–03 düzeltme tasarımı / ADR-0007 Proposed

- Hedef/ölçek/batch quantile kusurları yeniden üretildi; 40 mevcut ML testi
  dış socket/joblib engelli geçti. Üretim ML kodu, testler ve manifestler değişmedi.
- ADR-0007 Proposed: terminal NaN, hedef ufkuna göre train/test purge, train-only
  bounds ve ortak dört-model scaler/inverse, tutarlı advanced state ve format1
  joblib. Sürümsüz artifact yeniden eğitim gerektirecek; henüz uygulanmadı.
- Seçenekler/etkiler/kabul ölçütleri/rollback tasarımı, ADR dizini, backlog,
  mimari/veri modeli, test stratejisi ve Boot handoff güncellendi. AGENTS mimari/
  veri modeli kapısında açık onay beklenir; genel “devam” bu öneriye özel onay değil.
  Commit/push, frontend/canlı sağlayıcı veya yeni dependency yok.

### 2026-11-06 — BL-001 ML yardımcıları testleri

- 9 ortak helper, 15 simple predictor ve 16 advanced feature/eğitim testi
  eklendi; bağımsız oran/lag/pencere/eşik/metrik beklentileri, gerçek sklearn
  fit/scaler ve açık estimator/target boundary doubles birbirinden ayrıldı.
- 80/80 geçerli transient mutant assertion failure verdi; bağımsız inceleme
  40 testi doğruladı. Birleşik 288 backend testi dış ağ engelli geçti; 57 Python
  compile, 39 belge/11 Boot bölümü ve bağlantı/ADR/diff kontrolleri başarılı.
- Native sıfır target kuyruğu, eğitim/inference ölçek uyumsuzluğu, future-dependent
  clipping ve diğer ML-04–08 kusurları ayrı reproducer'larla backlog'a kaydedildi.
  Yanlış sonuçlar doğru beklenti sayılmadı; üretim ML kodu/manifestler değişmedi.
- Test ortamına mevcut gereksinimden sklearn eklendi; destek/pin kararı, yeni
  ADR, commit/push yok. Frontend/canlı sağlayıcı koşulmadı; genel ML doğruluğu,
  kalibrasyon ve M1 kapanışı açık. Ayrıntı `2026-11-06-ml-helper-tests.md` kaydında.

### 2026-11-06 — ADR-0006 onaylı OTT sözleşmesi

- Kullanıcı “onaylıyorum” dedi; ADR-0006 Accepted ve uygulandı. OTT ilk fiyatı
  loop öncesinde seed eder; `ott_trend` ilk gözlemde eksik, sonra ±1. Mevcut
  VAR/config/max-min recurrence/eşitlik ve puan ağırlıkları korunur.
- Boş/tek gözlem sözleşmesi, pozitif/sonlu Close doğrulaması ve mutasyondan
  önce ValueError eklendi. Eski ott/ott_signal alanları ve caller veri korunur.
- 10 contract + 6 gerçek entegrasyon testi, 18 kontrollü mutant ve bağımsız
  inceleme ile fiyat/yön/okuyucu zinciri doğrulandı. Temporal fiyat edge'i ayrıca
  RED→GREEN düzeltildi. Bu, canonical OTT veya finansal etkinlik kanıtı değildir.
- Veri modeli/backlog/test stratejisi/handoff güncellendi; RP-03 onaylı kapsamı
  kapandı. Genel M1 açık; canlı sağlayıcı/frontend tekrar koşulmadı. Commit/push,
  yeni API/dependency/kalıcı model/migration yok.
  Son birleşik 248 backend testi dış ağ engelli geçti; bağımsız incelemede açık bulgu yok.
  54 Python compile, 39 belge/11 Boot bölümü ve link/ADR/diff kontrolleri geçti.

### 2026-11-06 — RP-01–05 dar hata düzeltmeleri

- Risk/pozisyon RSI/MACD sıfırları mevcut kurallara girer; None/NaN atlanır.
  GÜÇLÜ SAT mevcut satış boyutunu kullanır; puan ve yüzdeler yeniden seçilmedi.
- DayTrader eksik veri/ATR, geçersiz entry fiyatı veya yönü ValueError ile
  hesap/kayıt mutasyonundan önce reddeder. Geçerli giriş/çıkış/P&L aynı;
  gerçek scan hatalı sembolü atlar ve devam eder.
- OTT pandas TypeError `.ffill()` ile çözüldü. Tam seed/ott_trend sözleşmesi
  ADR-0006 Proposed hazırlandı; henüz uygulanmadı. Yeni alan için açık onay bekler.
- 20 regresyon RED→GREEN, 94 hedef ve birleşik 232 dış ağ engelli test başarılı.
  Bağımsız inceleme bulgusuz. Dependency/API/kalıcı model değişmedi; commit/push
  yok. Frontend/canlı servis yeniden koşulmadı; genel M1/OTT kapsamı açık.
  52 Python compile, 39 zorunlu belge/11 Boot bölümü ve bağlantı/ADR/diff kontrolleri geçti.

### 2026-11-06 — BL-001 risk ve pozisyon test dilimi

- 13 risk, 16 pozisyon ve 17 DayTrader testi eklendi. Puan eşikleri, volatilite,
  stop/hedef/ATR/entry/R:R, confidence ve yerel yön/miktar P&L gerçek hesaplarla
  ve ayrı açık kural girdileriyle doğrulandı. 104 geçerli mutant yakalandı.
- Ayrı reproducer'lar RSI/MACD sıfır atlama, güçlü SAT boyut etiketi, OTT pandas
  3.0.1 TypeError/alan uyumsuzluğu ve doğrudan kısa/geçersiz girdi açıklarını
  doğruladı. Bunlar düzeltilmedi veya doğru beklenti olarak testlere yazılmadı;
  BACKLOG RP-01–05 ve handoff'ta açık kaldı. UI/canlı erişim iddiası yapılmaz.
- Üretim formülleri/config/API/model/dependency manifestleri değişmedi; yeni
  ADR veya commit/push yok. Genel risk/pozisyon/BL-001/M1 kapanışı iddia edilmez.
  Son birleşik 212 test dış ağ engelli geçti; bağımsız inceleme 46 hedef testi
  doğruladı. 49 Python compile ve belge/bağlantı/diff kontrolleri geçti.

### 2026-11-06 — BL-001 BOS ve kombinasyon testleri

- 15 BOS + 12 kombinasyon testi eklendi. Gerçek OHLCV zincirleriyle iki yönlü
  kırılma, swing/hacim/güç, kapanış/wick, lookback ve gerçek analyzer çıktıları;
  ayrı matcher testleriyle FVG+OB/FVG+BOS eşikleri ve confidence sırası doğrulandı.
- 40/40 geçerli bellek içi mutant assertion failure üretti, error yok; gerçek
  kodla 27 hedef ve birleşik 166 test dış ağ engelli geçti. Üretim hesapları,
  config/API/model/dependency manifestleri değişmedi; yeni ADR ve commit/push yok.
- Bağımsız incelemede önemli/engelleyici bulgu yok; tarama sınırı metni düzeltildi:
  sondan 10. mum dahil, daha yeni 9 mum hariç. Batch/future-neighbor ve sezgisel
  güç/confidence sınırları kaydedildi; canlı strateji veya kârlılık iddiası yok.
  Genel BL-001/M1 ve risk/pozisyon/ML kapsamı açık; frontend/canlı servis koşulmadı.
  46 Python compile, 39 zorunlu belge/11 Boot bölümü ve bağlantı/diff kontrolleri geçti.

### 2026-11-06 — BL-001 MACD/Bollinger değişken seri testleri

- 5 gerçek hesaplama testi eklendi: iki yönlü MACD şoku/reversion, ayrı sinyal/
  histogram değerleri ve warmup sınırları; Bollinger 20 fiyat penceresi,
  popülasyon std ve aykırı değer çıkışı. Girdi/index korunması kapsandı.
- Beklenen değerler bağımsız matematikten türetildi; 12/12 bellek içi hatalı
  varyant assertion failure verdi, gerçek kodla 17 hedef test yeniden geçti.
  Birleşik 139 test dış ağ engelli geçti. Üretim hesapları, config ve dependency
  manifestleri değişmedi; yeni ADR yok, commit/push yapılmadı.
- BACKLOG, TEST_STRATEGY, mimari test sayısı ve Project Boot handoff güncellendi.
  Bağımsız inceleme bulgusuz; 44 Python compile, belge/bağlantı/diff kontrolleri geçti.
  Frontend/canlı sağlayıcı doğrulaması yeniden koşulmadı; genel BL-001/M1 açık.

### 2026-11-05 — ADR-0005 onaylı RSS bütçesi ve kaynak tanıları

- Somut onay sorusuna “devam edelim” yanıtı kaydedildi; ADR-0005 Accepted.
  RSS requests byte fetch → feedparser yoluna taşındı; connect/read 8/8,
  en fazla 2 ek deneme, 1.5/3.0 bekleme, seçili retry nedenleri ve 429/503
  Retry-After ertelemesi uygulandı. Son HTTP hata gövdesi parse edilmez;
  parser hatası retry tetiklemez; yanıtlar kapanır ve kaynaklar bağımsız sürer.
- RSS/Yahoo instance tanıları kaynak/durum/deneme/monotonic süre/hata kodu tutar;
  her fetch yenilenir, haber özeti bağımsız kopya alır. Ham hata içeriği taşınmaz.
  Legacy özet/listesi kendi kaynak durumunu ve onaylı kısmi kesinti uyarısını gösterir.
- 22 RSS + 4 Yahoo + 5 legacy testi eklendi; 134 birleşik test dış ağ engelli
  geçti. Önceki RSS/NLP/hata assertion'ları yeni transport fixture'larına taşındı.
  Bağımsız inceleme bulgusuz; 44 Python compile ve belge/bağlantı/diff kontrolleri geçti.
  Yeni endpoint, kalıcı şema, dependency manifesti veya frontend değişikliği yok.
  Canlı sağlayıcı ve frontend build/UI yeniden koşulmadı; toplam hard deadline,
  global rate limit veya üretim gözlem kabiliyeti iddia edilmez. Genel M1 açık.

### 2026-11-05 — BL-003 hata yolları ve timeout iletimi

- Provider'ın mevcut `timeout_seconds` alanı gerçek Yahoo history çağrısına
  taşındı; legacy override yokluğu ve 8/2/1.5 varsayılanları korundu. History
  timeout'u toplam işlem/SDK deadline'ı olarak sunulmaz.
- RSS'in görünür HTTP >=400 hata gövdesindeki entry'leri haber sayması ayrı
  red→green düzeltildi; bozo ve 301 metadata uyumu/kaynak izolasyonu korundu.
- 6 provider, 5 gerçek adaptör ve 10 RSS kesinti testi eklendi. 24 hedef ve
  birleşik 103 test dış socket engelli geçti; 43 Python sözdizimi kontrolü başarılı.
  Bağımsız incelemede önemli/engelleyici bulgu yok. Frontend veya canlı sağlayıcı
  doğrulaması bu backend diliminde yeniden koşulmadı; manifest değişmedi.
- INTEGRATIONS/OBSERVABILITY/API_CONVENTIONS mevcut/hedef çelişkileri düzeltildi.
  Yeni RSS timeout/retry, geçici tanı ve kaynak uyarısı ADR-0005 Proposed olarak
  hazırlandı; açık onay bekler. Yeni politika henüz uygulanmadı; BL-003/M1 açık.

### 2026-11-05 — BL-002 onaylı uygulama

- Kullanıcı A metinleri ve ADR-0004 sorusuna “devam et” dedi; onay kaydedildi.
  Cockpit kaynak/nitelik/zamanı gösterir, yatırım uyarısı ve doğru boş/hata
  metinlerini kullanır; eski/sembol/oturum yanıtları güncel sonucu geri getirmez.
- RSS yayın/çekilme zamanı ayrıldı; bilinmeyen tarih bugünün saati olmaz. NLP
  hatası/nonfinite skor nullable/failed, gerçek sıfır nötrdür; özet ve filtreler
  yalnız hesaplanmış skorları kullanır. Gerçek skorsuz haber listede kalır.
- Legacy haber/ML metinleri güncellendi. Sabit olasılıklar kaldırıldı; geçersiz
  yayılımda ±%5 fallback senaryo/bant ve türetilmiş uyum/risk kartı gösterilmez.
  Geçerli tahmin/sinyal formülleri, ufuk ve sentiment eşikleri değişmedi.
- 14 RSS/NLP ve 8 kontrollü Streamlit AppTest ile toplam 82 test dış ağ engelli
  geçti; 10/10 üretim cockpit senaryosu ve Next.js build başarılı. Node/Python
  sözdizimi (41 Python dosyası), bağımsız inceleme ve diff kontrolü tamamlandı.
  İnceleme sonrası bozuk metadata tipi red→green ile düzeltildi.
- Mevcut requirements paketleri yalnız yerel `.venv` içine kuruldu; manifest
  veya dependency politikası değişmedi. Canlı sağlayıcı/tam legacy browser
  koşulmadı; Next.js/deprecation ve CDSK validator ortam sınırları sürüyor.
  Genel M1/M2 kapanmadı; commit/push yapılmadı.

### 2026-11-05 — BL-002 incelemesi ve onay paketi

- Cockpit ve legacy haber/ML kaynakları incelendi; analiz metadata'sının
  görünmemesi/eski analiz, gerçek zaman iddiası, sezgisel ML uyum ve görsel bant
  sunumu için kesin ekran metinleri Proposed belgeye alındı.
- Aktif haber ekranının gerçek RSS kullandığı ve mock yardımcılar için diğer
  Python modüllerinde doğrudan çağrı bulunmadığı kaynak/AST ile doğrulandı.
- RSS yayın tarihi eksik/geçersizse bugünün zamanının yazılması ve NLP hatasının
  nötr skorla birleşmesi için ADR-0004 Proposed hazırlandı; nullable skor/zaman
  sözleşmesi açık onay bekliyor. ADR indeksi, AI/veri modeli notları, backlog ve
  handoff güncellendi. Üretim kodu veya bağımlılık değişmedi; BL-002/M1 açık.
- Bağımsız incelemede bulunan sabit ML senaryo olasılıkları öneride giderilecek
  kapsam olarak eklendi; geçersiz yayılım, tahmin ufku ve sentiment eşik borçları
  görünür kılındı. Bu kayıtlar uygulanmış hesaplama düzeltmesi değildir.

### 2026-11-05 — Deterministik cockpit tarayıcı senaryoları

- Mevcut UI smoke'a ayrı `--deterministic` modu ve test route fixture'ları
  eklendi. Beş senaryo auth/watchlist/analiz/paper/RSI başarı, boş/kısmi/veri
  yokluğu, iki yönlü RSI tercih kalıcılığı ve 503 sonrası yeniden denemeyi sınar.
- Gerçek Next.js ekranı kullanılır; API/sağlayıcı yanıtları test içindedir.
  Beklenmeyen API/dış istek ve eksik Playwright deterministik modu başarısız kılar.
  Canlı smoke'un yenileme sonrası response bekleyicisi tıklamanın önüne alındı.
- `make smoke-ui-deterministic`, isteğe bağlı browser channel ve README komutları
  eklendi. Üretim UI/API ve dependency bildirimleri değişmedi; yeni ADR yok.
- Doğrulama: Playwright 1.62.1/headless Edge ile 5/5 senaryo dev ve üretim
  ekranlarında, Next.js üretim derlemesi ve iki Node sözdizimi kontrolü geçti.
  Bozulmuş tercih kaydı test tarafından yakalandı; eksik Playwright exit 1 verdi.
  Bağımsız incelemede engelleyici bulgu yok. Canlı sağlayıcı smoke'u ve 60 backend
  testi bu frontend adımında yeniden koşulmadı; M1 açık. Next.js kurulum güvenlik
  uyarısı ve tam CDSK validator'ın ortamda bulunmaması devam ediyor.

### 2026-11-05 — M1 patern/sinyal doğrulaması ve devam

- Kullanıcının devam talebiyle bekleme kaldırıldı; BL-001 için 14 patern ve
  11 sinyal testi eklendi. Sentetik mum/tepe/dip, FVG/dolum, order block,
  RSI/kesişim/trend, Bollinger/hacim ve çoğunluk kuralları kapsandı.
- 60 backend testi normal ve dış socket bağlantıları engellenmiş koşuda geçti;
  Windows asyncio için loopback bağlantıları serbest bırakıldı. Dört bellek içi
  hatalı hesaplama varyantı yeni assertion'larla yakalandı.
- 39 Python dosyasının sözdizimi ve 39 CDSK zorunlu dosyası/11 Boot bölümü
  yapısal kontrolü geçti. Belgelenen upstream CDSK validator bu Windows
  ortamında yok; frontend build/canlı UI smoke çalıştırılmadı.
- Üretim hesaplamaları, şema ve dependency bildirimleri değişmedi. Yerel test
  ortamı Python 3.12.14 ile hazırlandı; Starlette/httpx deprecation uyarısı açık.
  Handoff/backlog güncellendi; sıradaki iş kontrollü tarayıcı senaryoları.
  BL-001 ve M1 tamamlanmış sayılmaz; yeni ADR alınmadı.

### 2026-11-05 — Git teslim hazırlığı

- Kullanıcı commit + push istedi; CDSK belgeleri, kalan iş planı ve mevcut
  API/cockpit/test/geliştirme araçları tek teslim kapsamına alındı.
- README'deki son boşluk düzeltildi; uygulama davranışı ve bağımlılık
  sürümleri değiştirilmedi. Üretilmiş dosyalar ve yerel secrets dışarıda tutulur.
- Teslim öncesi 35 backend testi, CDSK doğrulaması ve UI smoke scripti sözdizimi
  kontrolü ve Next.js üretim derlemesi geçti. Next.js 14.2.18 için kurulum
  güvenlik uyarısı devam ediyor;
  bu bir üretim release'i veya güvenlik riskinin kapatılması değildir.

### 2026-11-05 — Bekleme ve kalan işlerin devri

- Kullanıcı isteğiyle geliştirmeye ara verildi; uygulama kodu değiştirilmedi.
- Execution backlog ve Project Boot handoff'a CDSK durumu, M1–M3 kalan işleri,
  kabul/onay kapıları ve yerel Git remote bilgisi yazıldı.
- Belgeler/yeni uygulama dosyaları için mevcut untracked durum görünür kılındı;
  commit/push yapılmadı ve uzak depo içeriği doğrulanmadı.

### 2026-11-05 — M1 hesaplama doğrulaması

- Gerçek `TechnicalAnalyzer` için 12 deterministik regresyon testi ve gerçek
  hesaplayıcılı 3 API testi eklendi; üretim hesaplamaları/dependencies değişmedi.
- EMA/SMA/VWMA, RSI yönü ve 22 dönemlik ısınma sınırı, karma kazanç/kayıp,
  MACD/Bollinger, destek-direnç, input koruması ve kısa veri uyarısı kapsandı.
- M1 milestone, mimari ve test stratejisindeki eski durum ifadeleri düzeltildi;
  backlog'a kalan beş iş, kabul ölçütleri ve onay sınırları eklendi.
- Doğrulama: 35 backend testi ve CDSK validator geçti. Python 3.9/LibreSSL
  uyarısı devam ediyor; frontend build ve canlı UI smoke bu değişiklik için
  yeniden çalıştırılmadı. M1/M2 tamamlandı veya üretime hazır iddiası yoktur.
- Testlerin ağsızlığı socket bağlantıları engellenerek, EMA/RSI testlerinin
  hata yakalama gücü bellekte yapılan iki kontrollü hesaplama değişikliğiyle
  doğrulandı (üretim dosyaları değiştirilmedi).

### Added

- CDSK 0.1.0 proje iskeleti ve ajan çalışma protokolü.
- Current-state ürün, mimari, execution, kalite, operasyon ve AI belgeleri.
- Önceliklendirilmiş güvenilirlik/ürünleştirme backlog'u.
- ADR-0001 (kapalı beta kapsamı), ADR-0002 (FastAPI/Next.js mimari),
  ADR-0003 (davet + magic link kimlik/bildirim) eklendi.
- `docs/10-architecture`, `docs/20-execution`, `docs/40-operations` ve
  proje özet belgeleri hedef mimari kararı ile güncellendi.
- FastAPI tabanlı ilk backend çekirdeği eklendi (`backend/app`): sağlık,
  davet + magic link auth, sembol ve izleme listesi, teknik analiz, RSI
  tarayıcı ve paper portföy endpointleri.
- Backend servis seviyesinde unit testler (`backend/tests/*`) eklendi (auth,
  watchlist, portfolio, response contract) ve API başlatma scripti
  (`run_api.py`) eklendi.
- Next.js MVP cockpit’u eklendi (`frontend/app/*`): demo auth, watchlist,
  teknik analiz, RSI tarama ve portföy önizleme akışlarını gösteren tek sayfa
  kullanıcı arayüzü.
- `frontend/package.json`, `frontend/package-lock.json` ve Next.js build desteği
  eklendi; `.next` üretim artifaktının repoya yazılmaması için `.gitignore`
  güncellendi.
- Tek komutla geliştirme başlatma scripti eklendi (`run_dev.py`) ile API ve
  Frontend aynı süreçte başlatılabiliyor; bağımlılık kontrolü ve health wait
  mantığı eklendi.
- `run_dev.py` için Windows çalıştırıcısı eklendi (`run_dev.bat`) ve frontend
  bağımlılığı eksik olduğunda isteğe bağlı otomatik `npm install` seçeneği eklendi.
- `run_dev.py` başlatma/sağlık kontrolü ve kapanış davranışı güçlendirildi;
  `--ready-timeout`, `--check-interval`, `--shutdown-timeout` seçenekleri eklendi.
- `scripts/run_dev_smoke.py` eklendi; `run_dev.py` için kısa süreli entegrasyon smoke
  kontrolü (hazır olma ve kontrollü kapanış) genişletildi; auth invite/magic,
  watchlist ve portfolio kontrat doğrulamaları da eklendi.
- `scripts/run_dev_smoke.py` ayrıca front-end kök sayfası HTML marker kontrolü ile
  cockpit doğrulamasını da kapsayacak şekilde genişletildi.
- Frontend smoke kontrolü, Next.js dev-mode chunk URL çıkarımı için esnek regex ve yedek
  `/_next/static/chunks/app/page.js` fallback ekleyerek, bölüm metinlerinin birden
  çok adımda doğrulanmasını sağlayacak şekilde sağlamlaştırıldı.
- Smoke test kapsamı workflow seviyesine taşındı; API tarafında watchlist için
  add/remove/set ve portfolio için add/re-add/remove akışları doğrulanarak temel
  uçtan-ucuna kritik kullanıcı iş akışları smoke’e dahil edildi.
- Tarayıcı tabanlı opsiyonel UI smoke senaryosu eklendi (`scripts/run_frontend_ui_smoke.mjs`).
- UI smoke senaryosunun kırılgan noktasını azaltmak için `frontend/app/page.js`
  tarafında test idleri (`data-testid`) eklendi.
- `scripts/run_frontend_ui_smoke_with_stack.py` eklendi; `run_dev.py` ile backend ve
  frontend’i birlikte başlatıp durduran ve Playwright UI smoke testini otomatik
  çalışan tek adımlı akışı sağlar.
- RSI screener API yanıtına `skipped_symbols` alanı eklendi; çekilemeyen semboller
  ayrı bir alanla dönüyor ve frontend’de kullanıcıya görünür uyarı olarak sunuluyor.
- `skipped_symbols` ile birlikte kontrat `warnings` alanını da getirir oldu; her
  atlanan sembole `symbol`, stabil `code` ve `reason` eklenerek nedeni gösterilir.
- RSI screener’de `warnings` etiketleri kullanıcı dostu görünüme kavuşturuldu;
  teknik `code` değerleri hala korunurken ekran metninde anlamlı Türkçe etiketler
  (`Veri alınamadı`, `RSI hesaplanamadı`) gösterilir.
- RSI `warnings` kartlarında görsel iyileştirme yapıldı; uyarılar ikonlu ve seviye
  rengiyle ayrıştırılarak listeleniyor (kritik/öneri ayrımı için temel tonlama).
- RSI uyarı paneline özet+detay akışı eklendi: atlanan toplam sembol sayısı,
  kod bazlı sayım etiketleri ve istemci tarafında detay aç/kapa kontrolü.
- Uyarı özet etiketleri, kullanım yoğunluğuna göre (sayıya göre) sıralanacak şekilde
  güncellendi; bu da aynı anda birden fazla kod türü geldiğinde okunabilirliği artırıyor.
- `run_dev.py` artık frontend başlatılırken `NEXT_PUBLIC_API_BASE` enjekte ediyor;
  frontend `page.js` bu değişkeni varsa onu öncelikli olarak kullanacak şekilde
  güncellendi.
- MIT lisans referansını destekleyen `LICENSE` dosyası eklendi.
- RSI screener ekranındaki sonuç tablosu (`frontend/app/page.js`) artık her satırda
  veri kaynağı (`source`), çekilme zamanı (`fetched_at`) ve yaşlanma bilgisi
  (`staleness_seconds`) göstermeye başlandı.
- Makefile eklendi; `make run-dev`, `make test`, `make smoke`, `make ci` ile
  tekrarlanabilir geliştirme ve doğrulama akışı oluşturuldu.
- GitHub Actions CI eklendi (`.github/workflows/ci.yml`) ve `push`/`pull_request`
  tetiklemeleriyle otomatik `make ci` akışı çalıştırılıyor.
- `backend/app` içindeki yanlış paket göreceli importları düzeltildi; bu sayede
  backend ve frontend birlikte smoke testleri başarılı oldu.
- `MarketDataProvider` test odaklı olarak fetcher enjeksiyonunu destekleyecek şekilde
  genişletildi ve bu kapsamda `backend/tests/test_market_data.py` eklendi.
- Offline deterministik API kontrat testleri eklendi: `backend/tests/test_api_contracts.py`.
- Aynı kontrat paketi içinde auth ve giriş doğrulama (401/400/422) senaryoları
  eklendi; meta alanları (`source`, `fetched_at`, `as_of`, `staleness_seconds`)
  doğrulaması genişletildi.
- RSI screener cevaplarına da veri kaynağı/uyarı meta alanları eklendi
  (`source`, `fetched_at`, `as_of`, `staleness_seconds`), böylece tarama sonuçları
  da izlenebilir hale geldi.

### Changed

- README'ye CDSK başlangıç yolu, prototip olgunluk notu ve venv tabanlı yerel
  başlangıç eklendi.
- Ignore politikası yerel venv, test/lint cache, coverage ve Streamlit secret
  dosyalarını kapsayacak şekilde genişletildi.
- ROADMAP, MILESTONES ve deployment/ops dokümantasyonu kapalı beta hedefi ve
  Hostinger/Coolify geçişiyle uyumlu hale getirildi.
- Dokümantasyonlarda Frontend Cockpit başlatma adımları ve backend+frontend
  paralel çalıştırma talimatları güncellendi.
- `CHANGELOG.md` ve `PROJECT_BOOT.md` güncellendi; `make ci` validasyonunun
  backend testleri + frontend build + smoke geçişini de kapsayacak şekilde kaydı
  güncellendi.
- Frontend smoke kontrolü, geliştirme orkestrasyonunda kullanıcı arayüzü ayağa
  kalkışını da doğrulayacak şekilde güncellendi.
- UI smoke senaryosu, `frontend/app/page.js` test id’lerini kullanacak şekilde
  daha kararlı hâle getirildi.
- `scripts/run_frontend_ui_smoke.mjs`, frontend bağımlılığına fallback yaparak
  Playwright modülünü `frontend/node_modules/playwright` konumundan da yükleyebilir.
- `make smoke-ui-stack` hedefi eklendi; stack başlatarak UI e2e smoke testini
  tek komutla çalıştırır.
- `make smoke-all` hedefi eklendi; `make smoke` ve `make smoke-ui-stack` ardışık
  çalıştırarak tek komutla uçtan-uca kapsamlı doğrulama sağlar.
- `make ci-e2e` hedefi eklendi (`make ci-core` + `make smoke-ui-stack`) ve geliştirmeye odaklı
  e2e kapsamını CI akışına bağladı.
- `make ci-full` hedefi eklendi (`make ci-e2e`) ve tam otomatik doğrulama zincirini
  tek komutta topladı.
- Yerel doğrulama playbook’u (`smoke`, `smoke-all`, `ci-e2e`, `ci-full`) netleştirildi.
- `make ci-core` hedefi eklendi; `make ci` akışını (deps/test/build+smoke) ve
  `make ci-e2e` akışını (`ci-core` + `smoke-ui-stack`) birbirinden bağımsız ve
  port çakışmasız hale getirmek için.
- `README` hızlı görevler listesine `make ci-core` eklendi; `ci-e2e`/`ci-full`
  akışının neyi içerdiği daha net hale getirildi.
- `make qa` hedefi eklendi; `ci-e2e` için günlük kapanışta okunabilir tek adımlık
  alias sağlandı.

### Validation

- CDSK validator geçti.
- 15 Python kaynak dosyasının sözdizimi kontrolü geçti.
- API katmanı (`backend/app/*.py`) için `py_compile` doğrulaması başarılı.
- Next.js uç kısmı için `npm install` ve `npm run build` ile derleme başarılı.
- Backend smoke testleri `python3 -m unittest discover -s backend/tests -p 'test_*.py'`
  ile **20 test** başarılı geçti.
- `run_dev.py --help` yeni opsiyonlarla birlikte doğrulandı; scriptin hata ve erken
  kapanış senaryoları manuel olarak test edildi.
- `python3 scripts/run_dev_smoke.py` ile kısa bir entegrasyon smoke koşusu doğrulandı
  (auth + watchlist add/remove/set ve portfolio upsert/re-upsert/delete workflow’ları +
  frontend HTML marker doğrulaması).
- `make smoke-ui` hedefiyle Playwright kurulu ise tarayıcı üzerinden temel UI etkileşim
  senaryoları çalıştırılabilir (opsiyonel).
- `make smoke-ui` senaryosu gerçek backend+frontend ortamında çalıştırılarak
  otomatik olarak doğrulandı.
- `make smoke-ui` hedefi çağrıldığında Playwright yüklü değilse açık kurulum
  yönergesi ile güvenli erken çıkış yapılır.
- `scripts/run_frontend_ui_smoke.mjs` için sözdizimi kontrolü (`node --check`) ve
  `make smoke-ui` davranışı doğrulandı.
- `scripts/run_frontend_ui_smoke_with_stack.py` için sözdizimi kontrolü
  (`python3 -m py_compile`) ve yeni `make smoke-ui-stack` hedefi doğrulandı.
- `scripts/run_frontend_ui_smoke.mjs` RSI tarama akışını da doğrulayacak şekilde
  güncellendi (`rsi-screen-button`, `rsi-results-table`) ve tablo başlıklarını kontrol
  edecek ek doğrulama eklendi.
- `scripts/run_frontend_ui_smoke.mjs` uyarı paneli varsa detay açma akışını da test eder
  hale geldi; panel detayları için selector tabanlı test kimlikleri eklendi.
- `scripts/run_frontend_ui_smoke.mjs`, RSI uyarı paneli detay görünürlüğünün
  kalıcılığını doğruluyor: paneli açtıktan sonra localStorage anahtarı
  (`trader_rsi_warning_details_visible`) beklenen değere ayarlanıyor mu kontrol
  ediliyor; yenileme sonrası panelin açık durumda kalması da doğrulanıyor.
- RSI uyarı paneli varsa, “Detayları göster” etkileşimiyle detay listesi renderı
  da Playwright UI smoke tarafında doğrulanabilir oldu (panel yoksa esnek davranış korunuyor).
- RSI screener endpointi, tek sembolde veri yoksa 500 döndürmek yerine o sembolü
  atlayıp kalan sonuçları döndürecek şekilde dayanıklılığı artırıldı.
- `make smoke-all` hedefi doğrulandı (`make smoke` + `make smoke-ui-stack` ardışık
  çalıştırması).
- `make ci` ile tam otomatik doğrulama koşusu başarıyla geçti (backend testler,
  frontend build ve run_dev smoke).
- `make ci-e2e` doğrulandı (`make ci-core` + `make smoke-ui-stack` ardışık koşusu).
- `make ci-full` doğrulandı (`make ci-e2e` ardışık koşusu).
- `make qa` doğrulandı (`make ci-e2e`).
- `backend/tests/test_api_contracts.py` içinde RSI screener sonuçları için kaynak
  ve zaman damgası meta alanlarının kontrat doğrulaması eklendi; test sayısı ve
  kalite kapsamı korundu.
- `backend/tests/test_api_contracts.py` içinde RSI için `insufficient_ohlcv_for_rsi`
  uyarı kodunu doğrulayan senaryo eklendi; böylece yetersiz teknik gösterge veri
  akışı da kontrat seviyesinde kapsandı.
- RSI uyarı panelindeki “Detayları göster/gizle” tercihi artık kalıcı hale
  getirildi; kullanıcı tercihi `localStorage` içinde saklanıp sonraki taramalarda
  korunur.
- Dokümantasyon tutarlılığı: `README`'deki lisans referansı artık gerçek
  `LICENSE` dosyasıyla eşleşiyor.
- Frontend `npm run build` doğrulandı; RSI screener tablosu, API’den gelen meta
  alanları render edecek biçimde render katmanıyla uyumlu hale getirildi.
- RSI screener kontratında `skipped_symbols`/`warnings` alanları ve frontend uyarısı
  da Playwright UI smoke senaryosuna eklendi; `warnings` artık standart kodlarla
  (`market_data_unavailable`, `insufficient_ohlcv_for_rsi`) dönüyor.
- RSI uyarı paneli için detay render elemanları `data-testid` ile etiketlendi ve
  test davranışını kırılgan selector aramalarına değil, kontrol edilen etkileşime
  dayandırarak stabilize edildi.
- RSI uyarı paneli, aynı akışta hem özet hem detay sunarak kullanıcıya
  daha odaklı bir tarama deneyimi sunacak şekilde güncellendi.
