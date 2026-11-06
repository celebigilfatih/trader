# ADR-0008 implementation plan

> **For agentic workers:** superpowers:executing-plans ve test-driven-development;
> native uygulama, sonunda fresh independent review.

**Goal:** Accepted tek ufuk cap'i, confidence input uyumu ve gözlem bazlı panel.
**Architecture:** MLPredictor public stateless bounds helper; mevcut fiyat response
ve legacy confidence çıktı alanları korunur. Streamlit paneli aynı helper'ı tüketir.
**Tech Stack:** mevcut unittest/sklearn/pandas/numpy/Streamlit AppTest/Plotly.
**Spec:** [Accepted ADR-0008](../50-decisions/ADR-0008.md).

## Global constraints

- Somut onay “onayladım, devam edelim”; yeni izin/onay kapısı yok.
- Başlangıç HEAD1aaed7b, clean codex/m1-pattern-signal-tests, upstream eşit.
  Mevcut çalışma konumu korunur; branch/worktree taşıma veya force-push yok.
- Önceki Commit & Push isteği kapsamında doğrulanmış bu devam işi aynı dala
  gönderilir; merge/PR/deployment yapılmaz. Git kimliği kendiliğinden değiştirilmez.
- Cap h1±.10,2–7±.25,8–30±.50,31+[-.70,1.00]; yalnız bir kez return clipping.
- Confidence optional current_price; fiyat/legacy kayıt yolları ayrı, mixed/invalid
  ValueError. Ensemble hariç; score/strict thresholds aynı, rounding yok.
- h gözlem sayısı; kesin hedef tarihi yok. Grafik0..h şematik adımlar; helper
  caption'ı tek kaynak. UI spread/signal, model/feature/purge/artifact1 aynı.
- Yeni dependency/calendar/vendor/API response alanı veya kullanıcı artifact işlemi yok.

## Review focus

- Ara h ve31+ sınırı, iki yön/eşitlik, çok büyük integer ufuk; fallback cap kalamaz.
- Positive finite current/price ile bile oran overflow; NaN/inf/bool/string/complex reject.
- Legacy extra alanlar korunur, invalid current her yolda reddedilir; boş/ensemble ayrı.
- Bayat veri ve ufuk selector değişimi: aynı gerçek last_index, takvim tarihi uydurulmaz.
- Chart anchor0 actual price, h endpoint model; band yalnız1..h, eski yayılım hesabı aynı.

## Task 1 — Bounds ve predict consumer

**Files:** modules/ml_predictor.py; Create backend/tests/test_ml_horizon_confidence.py.
**Interface:** get_prediction_return_bounds(prediction_horizon)->(float,float),
predict_price mevcut fiyat/error sözlüğü.

- [x] Literal helper interval/invalid/huge h ve gerçek scaler ile explicit model
  boundary iki yön/eşitlik/skip/ensemble/current-price testleri RED.
- [x] Public helper mevcut horizon validator'ı kullanır; predict inverse ardından
  bir kez np.clip uygular, ikinci fallback kaldırılır. Green hedef testler.

## Task 2 — Confidence uyumu

**Files:** modules/ml_predictor.py, backend/tests/test_ml_horizon_confidence.py.
**Interface:** get_model_confidence(predictions,current_price=None)->eski skor dict.

- [x] Price/legacy literal .7/.4/zero/extreme, caller korunması, missing context,
  mixed/error/empty/ensemble/type/nonfinite ve division overflow RED.
- [x] Girdileri mutasyonsuz doğrula, price/current-1; |r|>=.1 için0 ile overflow'u
  engelle, diğerlerinde eski formül. Legacy ek alanlar yok sayılır. Green tests.

## Task 3 — Gerçek legacy render

**Files:** app.py; backend/tests/test_legacy_disclosure.py fixture/expectation;
Create backend/tests/test_legacy_ml_horizon.py.

- [x] Mevcut FixtureML bounds method'u gerçek helper'a delegate eder; eski IO
  kontrolü sürer. Yeni selector/card/caption/stale-date/chart testleri RED.
- [x] Onaylı copy, h observation, actual last_index ve0..h line. Eski date-history
  trace yerine0 anchor (Son veri fiyatı), line0..h, band1..h; yeni forecast iddiası yok.
- [x] Eski numeric endpoint/band, model uyumu/risk/error testleri korunur;
  controlled AppTest green (monolith/browser/live-provider testi değildir).

## Task 4 — İnceleme, doğrulama ve teslim

- [x] Fresh reviewer scope source/independent oracles/ADR; önemli bulgu RED→GREEN.
- [x] Yeni testler için gerçekçi transient mutations/restore; tüm backend dış ağ
  engelli (asyncio loopback serbest), memory compile/CDSK39/Boot11/link/diff.
- [x] ADR index, disclosure doc, AI/DATA_MODEL/BACKLOG/TEST_STRATEGY/CHANGELOG/
  Boot güncel; scope risks/deferred UI spread calibration açıkça ayrı.
- [ ] Onaylı commit/push aynı dal; uzak SHA/clean/ahead-behind doğrulaması.

## Execution ledger

- Pre-flight Task1 helper→Task3 caption: aynı public signature ve numeric bounds.
- Pre-flight Task2 UI consumer yok; UI model-yayılımı hesabı değiştirilmez.
- Ruling: Grafik geçmiş timestamp trace'i yerine0 actual anchor kullanır;
  ADR0..h eksenini ve actual/current ile forecast ayrımını karşılar, band1..h kalır.
- Başlangıç331 backend/83 ML testi,60 Python dosyası; önceki UI10/build green.
  Bu dilimde Next.js kaynakları değişmediğinden frontend/build tekrar planlanmadı.


## Sonuç / doğrulama

- Task1/2 complete:14 yeni core testi önce9 assertion/subtest failure ve61 eksik
  API/type sözleşme hatası; helper+contract ile42 hedef test green (1.339 s).
  Yeni public helper eski validator'ı kullanır; tek clip ve aynı fiyat sözlüğü.
- Task3 complete:4 yeni panel testi önce7 assertion failure/error0; mevcut13 ile
  17 AppTest dış socket engelli green (7.103 s). Source AST ile gerçek fonksiyon
  render edilir; canlı monolith/browser/provider veya gerçek ML fit testi değildir.
- Ruling: FixtureML artık gerçek response gibi ensemble=110 taşır;108/112/110
  için std=sqrt(8/3), mean110. Eski fixture'ın ensemble eksikliğine bağlı112/108
  band beklentileri111.63/108.37 ve literal sqrt(8/3) ile düzeltildi. Production
  dispersion/signal/risk formülü değiştirilmedi; helper bu skora bağlanmadı.
- Bağımsız reviewer59 hedef testi yeniden geçti (7.610 s), önemli bulgu yok.
  Stale test açıklaması yeni legacy/price ayrımına göre düzeltildi.
- 16 core+3 actual-panel transient mutant assertion failure verdi; error yok.
  Cap uçları, tek clip/shared helper, ratio/abs/return units/strict thresholds,
  bool/finite/ensemble/error guard; panel adım kayması/ilk index/yanlış caption
  ufku hedeflendi. İlk reserved-error mutantı kaçtı;{"error":103.} eklendi ve
  yakalandı. Restore14 core/4 UI green; diskte mutation yok, genel coverage iddiası yok.
- Son birleşik349 backend testi dış connect/connect_ex engelli geçti (13.687 s);
  Windows asyncio loopback serbest, create_connection engelli.97 ML core,
  17 controlled Streamlit dahil. Önceki Starlette/Streamlit deprecation'ları sürer.
- 62 Python memory compile,39 zorunlu belge/11 Boot bölümü, Accepted ADR1–8,
  48 yerel doc link/whitespace/diff kontrolü başarılı. Simple/config/dependency/
  frontend aynı. Upstream macOS CDSK validator yolu yok; yapı onun sonucu değildir.
- Disclosure spec, AI/DATA_MODEL/OVERVIEW, BACKLOG, TEST_STRATEGY, ADR index,
  CHANGELOG ve Boot1.50 güncel.18 yeni test; M1 genel kapanışı/accuracy/kalibrasyon,
  UI dispersion tercihi ve işletim kapıları açık. Kullanıcı artifact'ı/migration veya
  yeni servis/dependency yok. Next.js/canlı sağlayıcı tekrarı yapılmadı.
- Başlangıç HEAD1aaed7b; kullanıcının önceki Commit & Push yetkisi korunur.
  Teslim commit'i ve uzak eşleşme onaylı Git adımı tamamlanınca kaydedilecek.
