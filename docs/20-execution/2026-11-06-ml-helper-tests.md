# BL-001 ML yardımcıları test planı

**Goal:** Yerel ML yardımcılarının desteklenen sayısal/veri hazırlama/eğitim
akışını bağımsız beklentilerle doğrulamak; yeni model veya finansal politika seçmemek.
**Architecture:** Mevcut SimpleMLPredictor/MLPredictor ve scikit-learn; sentetik
veri, gerçek helper/estimators ve açık estimator-boundary doubles ayrı tanımlanır.
**Tech Stack:** unittest/pandas/numpy, requirements'taki scikit-learn/joblib.
**Spec:** BACKLOG BL-001, AI_ARCHITECTURE guardrail ve TEST_STRATEGY P0 çekirdek.

## Sınırlar / review odağı

- Başlangıç HEAD `6ed1a59`, aynı dal. Önceki değişiklikler korunur; commit/push,
  üretim kodu/config/API/model/dependency manifesti değişikliği yok.
- Eksik scikit-learn yalnız `.venv` içine mevcut `>=1.3.0` gereksinimiyle kurulur;
  destek matrisi veya sürüm pinleme kararı değildir. Ağ/SMTP/gerçek piyasa ve model
  dosyası I/O birim testlerden ayrıdır. Ağsız inference/eğitim sentetik veridir.
- Beklentiler helper çıktısından türetilmez: literal/elle hesaplı oran, lag,
  pencere ve metrikler. RandomForest testi tam tahmin dizisini pinlemek yerine
  sabit hedef/seed veya anlamlı yaşam döngüsü ve sayısal ilişkiyi doğrular.
- Kural/estimator doubles gerçek sklearn hesaplama testi diye sunulmaz; gerçek
  fit/predict örneği ayrıca bulunur. NaN/inf/eksik veri, target ufku, chronological
  split, scaler ve özelliğin adı/sırası, hata izolasyonu review odağıdır.
- Şüpheli hedef kuyruğu, scaler uyumsuzluğu, lookahead temizlik, ufuk sınırı veya
  stale training durumu ayrı reproducer ile raporlanır. Kötü davranış golden
  assertion olarak onaylanmaz; bu test adımında sessizce model değişikliği yapılmaz.
- Testler mevcut davranışı karakterize eder; eksik feature RED'i iddia edilmez.
  Gerçekçi transient mutantlar yeni assertion'ları doğrular, diskte mutation yok.
- Başarı olasılığı, kârlılık, kalibrasyon veya üretim ML doğruluğu iddia edilmez.

## Task 1 — Ortak sayısal yardımcılar (ana ajan)

**Create:** [test_ml_helpers.py](../../backend/tests/test_ml_helpers.py).

- [x] Gerçek safe_divide sıfır/NaN/inf/özel fallback, clean_features clip/ffill/
  percentile ve tanılar, geçerli target prefix'inin ufuk/formülü/index'i.
- [x] Confidence helper'ın kendi desteklenen girdi sınırı ve eşikleri; predict
  ön-eğitim hatası. Yardımcı girdi/çıkış korunması ve empty sabit kolonlar.
- [x] Hedef testler ve mutant doğrulaması; target kuyruğu/uyuşmayan predictor
  dönüşü gibi şüpheler reproducer olarak açık kaydedilir.

## Task 2 — Simple predictor (simple worker)

**Create:** [test_simple_ml_predictor.py](../../backend/tests/test_simple_ml_predictor.py).

- [x] Gerçek prepare_features ratio/lag/rolling/optional indikatörler/temizlik,
  input/index korunması. Gerçek eğitimde 30 geçerli hedef sınırı ve zaman sırası.
- [x] Gerçek deterministic fit/predict zinciri; consumer-boundary model girdileri
  gerekiyorsa ayrı. Fiyat/getiri/yüzde/signal/confidence hesapları ve hata yolları.
- [x] Yalnız doğru desteklenen davranışlar için test; şüpheli feature-importance
  veya stale durum ayrı. Mutantlarla assertion gücü ve son green kaydı.

## Task 3 — Çoklu model features/eğitim (ML worker)

**Create:** [test_ml_features_training.py](../../backend/tests/test_ml_features_training.py).

- [x] Gerçek prepare_features: hand-checked indikator/lag/price-position örnekleri,
  bilinen clipping etkisi, optional seri sınırları ve caller korunması.
- [x] Gerçek scaler/estimator ile bir eğitim zinciri; chronological split/scaler
  fit sınırı, minimum veri ve hata izolasyonu. Hızlı gerçek estimator subset'i
  veya açık fit-boundary doubles gerekçesi kaydedilir; sklearn formülü mock'lanmaz.
- [x] Bilinen şüpheli terminal target/scaler/state çıktıları onaylanmadan ayrı
  raporlanır; yeni özellik şeması/eğitim politikası veya dependency seçilmez.

## Task 4 — Birleşim/inceleme/kalıcı kayıt

- [x] Tüm backend dış socket engelli (asyncio loopback serbest), yeni tests/
  oracle'lara bağımsız review. Kalan üretim kusurları ve test sınırları dürüstçe ayrılır.
- [x] Python compile, CDSK 39 belge/11 Boot bölümü, link/ADR/whitespace/diff,
  ML üretim/manifest değişmezliği; upstream validator ortam yokluğu ayrı yazılır.
- [x] AI_ARCHITECTURE, BACKLOG, TEST_STRATEGY, OVERVIEW, CHANGELOG ve Boot
  güncel; genel BL-001/M1/M2 tamamlandı denmez. Frontend/canlı servis koşulmaz.

## Başlangıç / kanıt

- Önceki 248 backend testi / 54 Python dosyası / 10 cockpit senaryosu kayıtlı.

## Sonuç / assertion kanıtı

- 9 helper + 15 simple + 16 advanced = 40 yeni test. Bu mevcut doğru davranışların
  test dilimidir; eksik üretim özelliği için RED→GREEN uygulaması iddia edilmez.
- Bellek içi function replacement ile 22 helper mutantı 37 assertion/subtest
  failure, 23 simple mutantı 53 failure; 35 advanced mutantı da assertion failure
  verdi. Son sayılan 80 mutantta error/survivor yok. Gerçek kod restore edilince
  testler geçti; üretim dosyaları diskte değiştirilmedi. Advanced ilk denemede
  ValueError üreten geçersiz mutant dışlandı, gerçekçi len+1 sınırıyla değiştirildi.
- Mutantlar ratio/sign/lag/window/ddof/optional uzunluk/default/clip/quantile,
  min30/min50/chronology/scaler/inverse/metrics/izolasyon ve confidence eşiklerini
  hedefler. Bunlar hata yakalama örnekleridir, genel mutation coverage metriği değildir.
- Simple gerçek RF50 fit/predict çalışır. Advanced gerçek feature+clean+scaler+
  LinearRegression altkümesi ve gerçek DummyRegressor nonzero hata baseline'ı
  çalışır: RMSE sqrt(.034490/12), MAE .0535, R² 1-34490/143. Hedefler bağımsız
  supplied labels+terminal NaN; native fill0 kusurunu doğrulamaz. Exact consumer
  eşikleri ve fit hataları açık boundary doubles'dır, gerçek model hesabı sayılmaz.
- Bağımsız reviewer 40 testi çalıştırdı (0.988 s), sayı/formül ve source sınırlarını
  kontrol etti; eyleme dönük bulgu yok. Ana ajan birleşik 288 testi çalıştırdı
  (6.892 s), dış connect/connect_ex ve create_connection engelli; yalnız Windows
  asyncio socketpair için loopback connect serbest. Gerçek Yahoo/RSS/SMTP yok.

## Ayrı runtime reproducer'ları — açık kusurlar

Aşağıdakiler istenen sözleşme/golden test değildir. Üretim ML davranışı aynı kaldı;
[BACKLOG ML-01–08](BACKLOG.md) ileride dar düzeltme ve gerektiğinde ADR/onay için kaydedildi.

1. **ML-01 hedef kuyruğu:** Close `[100,110,121,133.1]`, h=2 ile native
   create_target_variable son iki satır `[0,0]` verir; bu fiyatların geleceği yok.
   h=1/2/3 geçerli prefix oranları ayrı test edilir; fabricated tail test edilmez.
2. **ML-02 dönüşüm farkı:** Aşağıdaki gerçek LR/gerçek scaler reproducer'ında
   label boundary sabit .025'tir. Close160 için inverse edilmiş getiri fiyatı164
   olurken predict_price linear_regression/ensemble160 döner. Train input scaled,
   LR inference raw; yalnız SVR target inverse uygular. Bu politika onayı değildir.

```python
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from modules.ml_predictor import MLPredictor

sys.stdout.reconfigure(encoding="utf-8")
close = np.arange(100., 161.)
data = pd.DataFrame({"Open": close-2, "High": close+5, "Low": close-5,
                     "Close": close, "Volume": 1000.+10*np.arange(61)})
target = pd.Series([.025]*60 + [np.nan], index=data.index)
predictor = MLPredictor()
predictor.models = {"linear_regression": LinearRegression()}
with patch.object(predictor, "create_target_variable", return_value=target):
    print(predictor.train_models(data, {}))
print(predictor.predict_price(data, {}))
```

3. **ML-03 batch/lookahead:** clean_features(raw `[0,10,20]`) ilk değer .2;
   sona10000 ekleyip temizleyince aynı eski satır .3. Quantile tüm sample'dan
   hesaplanır; train_models hem feature hazırlığında hem split öncesinde temizler.
   Sonuç causal eğitim/backtest veya finansal etki ölçümü değildir.
4. **ML-04 state:** Başarılı fit'ten sonra simple'a yetersiz veriyle retrain
   error döner ama is_trained true kalır. Advanced başarılı LR fit ardından yalnız
   fit'te ValueError atan modelle retrain error döndürürken is_trained true ve
   trained_models eski linear_regression kaydını korur. Fresh-failure izolasyon
   testleri bu stale state kusurunun kapandığı anlamına gelmez.
5. **ML-05 adlar:** Simple yalnız 10 baz kolonla fit edilince sabit importance
   listesi RSI/EMA gibi olmayan özellikler adlandırır; son lag/volatility adları
   kaybolur. Aktif Simple UI kullanımı kanıtlanmadı, düzeltme yapılmadı.
6. **ML-06 NaN:** Simple estimator-boundary predict `[NaN]` döndürürse sonuçtaki
   fiyat/getiri NaN, signal HOLD, confidence .2. Sentetik sınır kusurudur; geçerli
   getiri assertion'ları bunu kapsıyor diye gösterilmez.
7. **ML-07 ufuk:** Advanced estimator-boundary .4 getiriyle h=1/3/7/14/30 için
   price/current 1.1/1.1/1.25/1.1/1.4 gözlendi. İkinci cap yalnız1/7/30'u map eder;
   app seçenekleri3/7/14/30. Beklenen yeni cap politikası seçilmedi.
8. **ML-08 interface:** get_model_confidence({'linear_regression':110.,
   'ensemble':110.}) TypeError: float object is not subscriptable. predict_price
   float fiyatlar döndürür; helper nested predicted_return bekler. Aktif consumer
   bulunmadı; helper'ın kendi desteklenen sınırı test edilir.

Windows varsayılan charmap stdout, predict_price Unicode uyarısını yazarken
hata sözlüğüne yol açtı; UTF-8 stdout ile yukarıdaki core reproducer çalışır.
Bu yerel gözlemin deployment/UI etkisi doğrulanmadı. Her iki modülün importta
warnings.filterwarnings('ignore') yan etkisi değişmedi.

## Ortam / kapanış

- Python3.12.14, numpy2.3.5, pandas3.0.1, sklearn1.9.1, SciPy1.18.1,
  joblib1.6.0; mevcut requirements sklearn>=1.3.0 ile yalnız ignored .venv'e
  kuruldu. SciPy cold importları ilk sefer uzun sürdü, sonunda tamamlandı;
  sonraki importlar ~3 s. Tanı için .venv/ml-compat altına kurulan SciPy1.16.3
  sys.path/testlere alınmadı. Ortam yeniden üretilebilir pin/support kararı değildir.
- 57 Python dosyası memory compile; 39 zorunlu belge/11 numaralı Boot bölümü,
  yerel belge bağlantıları, Accepted ADR-0004/5/6, whitespace/diff ve ML üretim/
  manifest değişmezliği kontrol edildi. Upstream macOS CDSK validator yolu burada
  yok; yapısal doğrulama upstream validator sonucu değildir.
- CHANGELOG, BACKLOG, AI_ARCHITECTURE, OVERVIEW, TEST_STRATEGY ve Boot1.45
  güncellendi. HEAD6ed1a59, commit/push yok; önceki dirty değişiklikler korundu.
  Canlı sağlayıcı/frontend tekrar koşulmadı. Dört varsayılan modelle tam native
  eğitim/inference, model save/load uyumu, accuracy/kalibrasyon/kârlılık ve genel
  BL-001/M1/M2 açık. Sonraki adım ML-01–03 için dar düzeltme planı; veri/feature/
  mimari sözleşmesi değişecekse somut Proposed ADR ve AGENTS açık onayı gerekir.
