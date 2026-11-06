# ADR-0007 ML correction implementation plan

> **For agentic workers:** Use superpowers:executing-plans and test-driven-development;
> implementation proceeds inline, with a fresh independent reviewer at completion.

**Goal:** Accepted ADR-0007 hedef/ön işleme/inference ve artifact sözleşmesini uygulamak.
**Architecture:** Mevcut MLPredictor içinde sabit feature hazırlığı, train-only bounds,
ortak scaler ve coherent fitted paket. Dört mevcut estimator/config aynı kalır.
**Tech Stack:** unittest, pandas/numpy, sklearn/joblib; mevcut ignored .venv.
**Spec:** [Accepted ADR-0007](../50-decisions/ADR-0007.md).

## Global constraints

- Onay: 2026-11-06 somut ADR-0007 sorusuna “Onaylıyorum.”; yeniden izin istenmez.
- HEAD6ed1a59, aynı codex dalı/dirty çalışma; taşıma/stash/commit/push yok.
  Worktree skill kontrolü mevcut devam bağlamını koruyarak uygulandı; .git'e yazılmaz.
- Kullanıcı belgelerinin CDSK yerleşimi tercih edilir; bu plan aynı zamanda kalıcı
  execution ledger'ıdır (shell tabanlı SDD helper'ları yerine Windows uyarlaması).
- Yeni bağımlılık/vendor/API/UI kapsamı yok; SimpleMLPredictor/cap/confidence aynı.
- h pozitif integer (bool hariç), min50 etiket; purge i+h<b, sonrasında train/test>=2.
- Sabit temizlik [-1,10]/ffill/zero; q01/q99 sadece purge edilmiş train'de bir kez.
- Format1: bounds ve eğitim ufku; sürümsüz dosya reddi/yeniden eğitim, kullanıcı dosyası yok.

## Review focus

- Gappy valid mask ve düzensiz tarih index'i: purge filtered offset değil orijinal konum.
- Sabit kolon ve nonfinite hedef: sıfır getiri ile bilinmeyen ayrılır, bounds uydurulmaz.
- Fit başarılı metric başarısız: model pakete giremez; başarısız retrain eski paketi kullanmaz.
- Inference extra/missing feature ve farklı ufuk: stored order, missing/mismatch error.
- Bozuk artifact alan/shape/fitted state: load önce doğrular; eski instance korunur.

## Task 1 — Target, split ve train-only bounds

**Files:** Modify modules/ml_predictor.py; Create backend/tests/test_ml_contract.py;
Modify backend/tests/test_ml_helpers.py, backend/tests/test_ml_features_training.py.
**Interfaces:** _validate_horizon(h)->int, _validate_data(data)->None,
_fit_clip_bounds(X_train)->dict, _apply_clip_bounds(frame,bounds)->DataFrame.

- [x] Literal target/NaN tail/real0, invalid horizon/price/index, native49/50,
  N53/h3→37/10 purge ve gappy original position, stateless prefix ve .2/19.8 bounds
  testlerini yaz; mevcut clip/split oracles'ını yeni Accepted sözleşmeye güncelle.
- [x] `.venv\Scripts\python.exe -B -m unittest backend.tests.test_ml_contract`
  ile RED'i izle; eksik davranıştan assertion/error beklenir, typo kabul edilmez.
- [x] Hedef, sabit cleaner, purge split ve bounds yardımcılarını uygula. X/y scaler
  fit'i yalnız tutulmuş train ile; mevcut feature formülleri/sırası korunur.
- [x] Hedef/feature/helper testleri green; yeni gerçek LR/scaler input oracles'ı doğrulanır.

## Task 2 — Shared inference ve coherent training state

**Files:** modules/ml_predictor.py, backend/tests/test_ml_contract.py.
**Interfaces:** clip_bounds:dict, prediction_horizon:int|None;
train_models mevcut metric/count sözlüğü; predict_price mevcut price/error sözlüğü.

- [x] Dört gerçek estimator + bağımsız .025 label boundary fiyat160→164; her model
  scaled input ve inverse target, native dört-model smoke, ufuk/kolon/index hata,
  fit/metric failure ve successful→failed retrain testleri RED.
- [x] Her model için stored bounds→X scaler→predict→y inverse→mevcut caps;
  yeni train denemesinde state reset, local fit/metrics başarılı modelleri birlikte publish.
- [x] Contract + korunmuş ML testleri green; independent targets ve native smoke ayrı raporlanır.

## Task 3 — Versioned trusted artifact

**Files:** modules/ml_predictor.py; Create backend/tests/test_ml_artifacts.py.
**Interfaces:** save_models(path)->None; load_models(path)->None/ValueError;
format_version=1 + preprocessing.clip_bounds + prediction_horizon + mevcut alanlar.

- [x] Save-before-fit, temp gerçek joblib roundtrip, legacy/unknown/corrupt fields/
  feature/bounds/scaler/model boyutu, load failure instance preservation testleri RED.
- [x] Paketi kaydetmeden/yüklemeden doğrula; load yalnız tam doğrulama ardından
  state atar. Kullanıcı artifact'ı açılmaz; otomatik migration veya güvenilmeyen yükleme yok.
- [x] Artifact ve tüm ML testleri green; gerçek temp I/O ile network engeli ayrıdır.

## Task 4 — Review, birleşim, kalıcı kayıt

- [x] Fresh independent reviewer: yalnız ADR-0007 ML patch/test/oracle kapsamı;
  Important/Critical bulgular RED→GREEN; accepted kapsam dışı bulgular ledger.
- [x] Tüm backend dış socket engelli (Windows asyncio loopback serbest); yeni
  davranışlar için gerçekçi transient mutations; Python memory compile/CDSK yapı/
  link/ADR/diff; frontend/canlı sağlayıcı yeniden koşulmaz.
- [x] AI_ARCHITECTURE, DATA_MODEL, OVERVIEW, BACKLOG, TEST_STRATEGY, ADR dizini,
  CHANGELOG ve Boot/handoff güncel; M1/accuracy kapanışı iddia edilmez.

## Execution ledger

- Pre-flight: Task1 bounds/feature/schema → Task2 aynı stored dönüşümler; uyumlu.
- Pre-flight: Task2 coherent state → Task3 yalnız tamamlanmış paket; uyumlu.
- Native uygulama seçimi mevcut açık uygulama onayı kapsamında ajan kararıdır;
  yeni onay kapısı eklenmedi. Komutlar Windows venv ile; .git/shell helper mutation yok.
- Başlangıç: 288 backend/40 ML testi, 57 Python dosyası. Sonuç aşağıdadır.


## Sonuç / doğrulama

- Task1 complete:19 contract testinde ilk29 assertion/subtest failure ve7 eksik
  sözleşme hatası; hedef/bounds/split/state ilk15 testi green. Mevcut helper ve
  training oracles Accepted stateless/purge kuralına uyarlandı; test zayıflatılmadı.
- Task2 complete: ilk59 ML testi green. Dört gerçek estimator ayrı .025 label
  boundary ile160→164; native hedef ve dört varsayılan estimator smoke'u ayrıca.
  Bu doğrulama finansal accuracy veya indicator causal garantisi değildir.
- Task3 complete: ilk9 artifact testinde sandbox temp izin hatası çevre engeliydi;
  geçerli sandbox dışı RED43 failure/7 eksik format hatası, sonra68 ML green.
  Gerçek trusted temp joblib roundtrip ve bad/legacy-load-before-state-change.
- Review: bağımsız68-test incelemesi P2 scaler coherence ve P3 dev ufuk buldu.
  Üç regresyon7 failure+1 OverflowError→green. İlk sayısal sample kontrolü kurulu
  sklearn1.9.1 StandardScaler'ın np.float64(61.0) sayısını reddetti (10 error);
  sonlu tam-değerli Real sayısı, bool hariç ve iki scaler eşitliğiyle düzeltildi.
  İki yeni artifact testiyle70 ML green; reviewer70 testi yeniden geçti (9.037 s),
  eyleme dönük bulgu kalmadı. Constant-price/zero-target paketi de ayrıca doğrulandı.
- Son kabul kontrolünde dtype belirtilmemiş empty Close için1 ValueError RED;
  helper float boş Series döndürerek düzeltildi, aynı test yeniden green ve
  reviewer tarafından bağımsız geçti. Ufuk doğrulaması boş girdide de korunur.
- 15/15 geçerli transient mutant assertion failure verdi, error yok. Target
  fill0/future-price mask, horizon bool, min50, purge equality/original position,
  q01/q99/clip, bounds fit kapsamı, reset, metric-failure publish, X scaler/target
  inverse ve horizon mismatch hedeflendi. İlk reset text eşleşmeyen varyant
  geçerli mutant sayılmadı; doğru kaynakla tekrar edilip yakalandı. Disk mutation yok.
- Task4 complete: son318 backend testi dış connect/connect_ex engelli geçti
  (19.869 s); yalnız Windows asyncio loopback serbest, create_connection engelli.
  Controlled13 Streamlit AppTest dahil; frontend/build ve canlı sağlayıcı tekrar yok.
  Starlette/httpx ve Streamlit deprecation uyarıları önceki borç olarak sürer.
- 59 Python memory compile,39 zorunlu belge/11 Boot bölümü, Accepted ADR1–7,
  yerel link/whitespace/diff kontrolleri geçti. CDSK upstream macOS validator yolu
  yok; bu yapısal kontrol upstream başarısı değildir. SimpleMLPredictor, config,
  dependency manifestleri ve estimator hiperparametreleri değişmedi.
- Güncel belgeler: ADR/index, AI_ARCHITECTURE, DATA_MODEL, OVERVIEW, BACKLOG,
  TEST_STRATEGY, RUNBOOK, CHANGELOG ve Boot1.47. HEAD6ed1a59, commit/push yok;
  kullanıcı/önceki dirty değişiklikler korundu. Kullanıcı artifact'ı açılmadı veya
  migrate edilmedi; legacy model yeniden eğitim ister. Simple ML-04–06 ve
  ML-07/08/takvim politikası ile genel BL-001/M1/M2 açık; sonraki adım Simple
  yaşam döngüsü/importance/sonlu tahmin dar düzeltmeleri.
