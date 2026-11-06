# Simple ML-04–06 düzeltme uygulama planı

> **For agentic workers:** superpowers:executing-plans ve test-driven-development;
> inline uygulama, sonunda fresh independent code review.

**Goal:** SimpleMLPredictor stale training/cache, importance isimleri ve nonfinite
tahmin kusurlarını mevcut dönüş biçimleriyle düzeltmek.
**Architecture:** Mevcut RF50/max_depth10/random_state42 ve hazırlama/eğitim akışı;
estimator'ın mevcut fitted feature_names_in_ metadata'sı kullanılır.
**Tech Stack:** Mevcut sklearn/pandas/numpy/unittest; yeni bağımlılık yok.
**Spec:** BACKLOG ML-04–06 ve PRODUCT_SPEC hata izolasyonu; ADR-0007 advanced
uygulaması korunur, bu adım o ADR'nin kapsamını genişletmez.

## Global constraints / karar

- Kullanıcının “Sıradaki adım” talebi, handoff'ta sıradaki bu dar hata düzeltmesini
  yetkilendirir. API/çıktı alanı, fiyat/etiket/confidence politikası, artifact formatı,
  yeni model veya mimari değişmez; yeni ADR/onay kapısı gerekmiyor.
- HEAD6ed1a59, aynı dirty codex dalı; önceki değişiklikler/çalışma konumu korunur,
  stash/worktree taşıma/commit/push yok. Windows'ta mevcut .venv kullanılır.
- Üretim değişikliği yalnız modules/simple_ml_predictor.py. Advanced module/config/
  dependency manifestleri başlangıç SHA256 ile karşılaştırılacak.
- Train başarı/hata, predict error ve importance DataFrame kolonları aynı kalır.
  Feature schema drift mevcut estimator doğrulamasıyla hata verir; yeni uyarlama yok.
- Forest fit etmiş kalsa da hata sonrasında sınıfın trained flag'i false olmalıdır;
  inference yeniden eğitim ister. Prediction hatası yalnız son sonuç cache'ini siler.
- Aktif Simple UI tüketimi kaynakta bulunmadı; gerçek uygulama ML paneli advanced
  sınıfı kullanır. Canlı/UI/finansal etkinlik iddiası yapılmaz.

## Review focus

- Başarılı fit/predict ardından short/preparation/fit/metric hatası eski state'i tutamaz.
- Sadece optional RSI/EMA/VWAP varsa isim/importance eşleşmesi gerçek fitted sıradadır.
- NaN/±inf getiri veya fiyat, taşan yüzde/fiyat ve pozitif olmayan fiyat sonuç değildir.
- Inference hata sonrası eski son tahmin gösterilemez; geçerli sonraki tahmin çalışır.
- Geçerli sıfır getiri ve mevcut ±.01 strict signal/confidence/percent dönüşümü korunur.

## Task 1 — RED regresyonlar

**Files:** Create backend/tests/test_simple_ml_fixes.py.
**Interfaces:** train_quick_model(data,indicators)->success/error dict,
predict_next_day(data,indicators)->price/error dict, get_feature_importance()->DataFrame.

- [x] Gerçek fit→failure/optional kolon importance ve gerçek refit yaşam döngüsünü
  test et. Expected labels literal baz/partial listeleri, sayısal önem eşleşmesi
  gerçek estimator importance tüketiminden; veri/caller korunur.
- [x] Explicit return-boundary double ile invalid return/current price/overflow ve
  eski cache temizliği test et; double forest fitting kanıtı sayılmaz.
- [x] `.venv\Scripts\python.exe -B -m unittest backend.tests.test_simple_ml_fixes`
  RED; başarısızlık nedeni eksik davranış olmalı. Yeni API nedeniyle typo hata yok.

## Task 2 — Küçük düzeltme / GREEN

**Files:** modules/simple_ml_predictor.py; existing tests korunur.

- [x] Her train girişinde is_trained=false/last_prediction=None; başarı flag'i
  yalnız sonlu metriklerden sonra. Her predict girişinde eski cache temizlenir.
- [x] Sonlu return/current price ve hesaplanan price/percent/change doğrulanır;
  pozitif olmayan current/predicted price hata. Hata mevcut error sınırında döner.
- [x] Hardcoded importance adları yerine model.feature_names_in_ kullanılır;
  metadata/sayı uyuşmazlığında mevcut boş DataFrame fallback korunur.
- [x] Yeni ve korunmuş simple/helper testleri green; gerçekçi transient mutation,
  restore ve production diff kontrolü. RF hyperparameter/feature formülü aynı.

## Task 3 — Review / birleşim / kalıcı kayıt

- [x] Bağımsız reviewer source/oracle/scope'u inceler; önemli bulgu RED→GREEN.
- [x] Tüm backend dış socket engelli (asyncio loopback serbest), Python memory
  compile, CDSK39 belge/11 Boot bölümü, ADR/link/whitespace/diff ve hash kontrolü.
- [x] BACKLOG, AI_ARCHITECTURE, OVERVIEW, TEST_STRATEGY, CHANGELOG ve Boot
  güncel. Upstream validator ortam eksikliği/canlı smoke/finansal doğruluk ayrıdır.

## Execution ledger

- Root cause: train_quick_model eski trained flag/cache'i sıfırlamıyor; importance
  sabit17 adını10 baz+optional sayıya kesiyor; predict nonfinite kontrolü yapmıyor.
- Pre-flight: ortak flag/cache ve RF fitted metadata; mevcut çıktı biçimleriyle
  çelişki yok. Yeni internal/persistent schema yok; native execution tercih edildi.
- Başlangıç318 backend/70 ML testi ve59 Python dosyası; sonuç aşağıdadır.


## Sonuç ve kanıt

- Task1 complete:13 yeni test eski kodda29 assertion/subtest failure, error0
  üretti. Short/preparation/fit/metric failure sonrası stale flag/cache, actual
  bare/partial feature isimleri ve nonfinite/overflow hataları yeniden üretildi.
- Task2 complete: mevcut15 simple +9 helper ile37 hedef test green (0.880 s).
  Gerçek RF fit/importance ve ayrı numeric return boundary birbirinden ayrıdır.
  Yeni internal schema yok; mevcut RF feature_names_in_ metadata'sı tüketilir.
- 11 geçerli bellek içi mutant11 assertion failure sonucu verdi (bazıları birden
  fazla subtest failure); error/survivor yok. Flag/cache reset ve publish, metric
  guard, hardcoded names/weight tersliği/duplicate metadata, percent overflow ve
  nonpositive predicted price kontrolü hedeflendi. Disk mutation yok; restore13
  test green (1.447 s). Bu örnekler genel mutation coverage metriği değildir.
- Task3 review complete: bağımsız reviewer37 testi yeniden geçti (0.877 s);
  source/oracle/partial optional/lifecycle/retry ve geçerli aritmetiği kontrol etti,
  eyleme dönük bulgu yok. Yeni review düzeltme turu gerekmedi.
- Son331 backend testi dış connect/connect_ex engelli geçti (11.437 s); Windows
  asyncio loopback serbest, create_connection engelli. Toplam83 ML testi; mevcut
  trusted-temp artifact ve13 kontrollü Streamlit testleri de bu birleşime dahildir.
- 60 Python memory compile,39 zorunlu belge/11 Boot bölümü, Accepted ADR1–7,
  yerel link/whitespace/diff kontrolleri ve advanced/config/dependency başlangıç
  SHA256 karşılaştırması başarılı. Son satır whitespace kalite kontrolünde
  temizlendi; davranış değişikliği değildir. Upstream macOS CDSK validator yolu
  yok; yapısal doğrulama upstream sonucu değildir. Starlette/Streamlit uyarıları sürer.
- BACKLOG/AI_ARCHITECTURE/OVERVIEW/TEST_STRATEGY/CHANGELOG/Boot1.48 güncel.
  Üretimde yalnız Simple modülü; yeni regression test dosyası ve bu kayıt eklendi.
  RF hyperparametreleri, feature/target/split/min30, geçerli sinyal/confidence ve
  mevcut error/success/importance alanları aynı. Advanced ADR-0007 kodu değişmedi.
- HEAD6ed1a59, commit/push veya kullanıcı artifact'ı işlemi yok; önceki dirty
  değişiklikler korundu. Yeni ADR/model/vendor/migration/çıktı sözleşmesi yok.
  Kaynakta Simple için aktif UI tüketimi bulunmadı; frontend/canlı sağlayıcı
  yeniden koşulmadı. ML-07/08/takvim, genel BL-001/M1/M2 ve finansal doğruluk/
  kalibrasyon ayrı açık; sıradaki adım kalan ML politika/sözleşmesini somutlaştırmak.
