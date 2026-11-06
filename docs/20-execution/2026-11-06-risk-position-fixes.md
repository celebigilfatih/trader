# RP-01–05 hata düzeltme planı

> Bu kayıt OTT ADR-0006 onayı öncesi dar düzeltme aşamasıdır. Kullanıcı sonradan
> “onaylıyorum” dedi; Accepted sözleşmenin uygulama kanıtı
> `2026-11-06-adr0006-implementation.md` içindedir. Aşağıdaki Proposed/onay
> bekliyor ifadeleri önceki aşamanın tarihsel kaydıdır.

**Goal:** Mevcut kuralları yanlış uygulayan koşulları düzeltmek, geçersiz
DayTrader girdisinin geçerli sonuç gibi ilerlemesini engellemek ve OTT için
kalan veri sözleşmesi kararını somutlaştırmak.
**Architecture:** Mevcut modüller; yeni servis/kalıcılık/API/dependency yok.
**Tech Stack:** unittest/pandas/numpy, mevcut yerel Python ortamı.
**Spec:** BACKLOG RP-01–05; `2026-11-06-risk-position-tests.md` reproducer'ları.

## Sınırlar ve karar ayrımı

- Başlangıç HEAD `6ed1a59`, aynı dal; önceki değişiklikler korunur. Commit/push yok.
- RP-01: geçerli sıfır ile None/NaN ayrılır; mevcut puan ağırlıkları ve eşikler
  değişmez. RP-02: güçlü satış etiketi mevcut satış yüzdesini kullanır, güçlü
  alış yüzdesine düşmez; yeni tahsis yüzdesi seçilmez.
- RP-04–05: private helper'lar geçersiz/yetersiz girdi için ValueError verir;
  geçerli BUY/SELL/HOLD matematiği ve kayıt formatı aynı kalır. Mevcut scan zaten
  sembol başına hatayı yakalar, `<50` barı atlar. Yeni HOLD/fallback fiyat yok.
- RP-03 runtime: eski `fillna(method='ffill')` yerine eşdeğer `.ffill()`;
  dependency yükseltme veya algoritma değişikliği değil. Ancak mevcut loop
  başlangıcı seed edilmeden okunuyor; sadece runtime düzeltmesi tam OTT çözümü değildir.
- OTT seed ve yeni `ott_trend` Series veri sözleşmesi ayrı Proposed ADR-0006'da
  somutlaştırılır; AGENTS veri modeli kapısı gereği Accepted sayılmaz/uygulanmaz.
- Review: sıfır/eksik/nonfinite göstergeler, satış yönü, geçersiz veri öncesinde
  atomik kayıt davranışı, gerçek scan izolasyonu, OTT dar uyumluluk iddiası.

## Task 1 — RP-01/02 (alert worker)

**Modify:** `modules/alert_system.py`.
**Create:** `backend/tests/test_risk_position_fixes.py`.

- [x] Önce RED: RSI0 ve MACD line/signal0 risk/pozisyon katkıları; None/NaN
  hesaplanmaz. Gerçek düşen seri RSI0'ı risk/pozisyon okuyucusuna taşır.
- [x] RED: GÜÇLÜ SAT mevcut SAT boyut metnini üretir; güçlü AL ve diğer mevcut
  boyut/puan dalları korunur. Yalnız ilgili guard/yön sırası düzeltilir.
- [x] Yeni ve önceki risk/pozisyon/sinyal testleri green; formül/ağırlıklar aynı.

## Task 2 — RP-04/05 (DayTrader worker)

**Modify:** `modules/day_trader.py`.
**Create:** `backend/tests/test_day_trader_validation.py`.

- [x] RED: boş/14 bardan kısa ATR ve entry, nonfinite/pozitif olmayan entry
  fiyatı veya ATR, desteklenmeyen position type. ATR sıfır hesap değeri olarak
  kalabilir; entry için pozitif ATR gerekir. Parametreye uygun yeterli veri korunur.
- [x] ValueError, anlamlı hesap/kayıt yan etkisinden önce. add ve close için
  yalnız BUY/SELL desteklenir; mevcut bozuk kayıt close'da değişmez. Olmayan
  sembol kapatma etkisizdir. Geçerli fiyat/miktar için yeni büyüklük kuralı seçilmez.
- [x] Gerçek scan bir ValueError veren sembolü atlar ve diğerine devam eder;
  geçerli entry/R:R ve BUY/SELL P&L eski testleriyle green.

## Task 3 — RP-03 runtime ve OTT karar paketi (ana ajan)

**Modify:** `modules/technical_analysis.py`.
**Create:** `backend/tests/test_ott_runtime.py`, `docs/50-decisions/ADR-0006.md`.

- [x] Gerçek add_indicator('ott') ile pandas TypeError RED; `.ffill()` sonrası
  runtime/index/girdi korunması green. Bu test doğru trend veya kârlılık kanıtı değil.
- [x] Önceki API'yle uyumlu kaynak/tekrar üretim: seed sonrası loop, tüm boş
  OTT kuyruğu ve okuyucu alan eksikliği kaydedilir. Yanlış sabit/sell sonuçlar
  doğru finansal beklenti olarak dondurulmaz.
- [x] Yeni seed/ott_trend sözleşmesi, sayı örnekleri ve rollout/rollback içeren
  Proposed ADR; kodda henüz seed veya yeni alan yok. Diğer işler için yeniden onay yok.

## Task 4 — Birleşim/kapanış

- [x] Tam backend dış ağ engelli (asyncio loopback serbest), bağımsız reviewer,
  Python compile/CDSK belge/link/diff ve manifest değişmezliği.
- [x] BACKLOG, TEST_STRATEGY, OVERVIEW, CHANGELOG ve PROJECT_BOOT güncel;
  çözülen RP-01/02/04/05 ile kısmi RP-03 açıkça ayrılır. M1/üretim kapanmaz.

## Kanıt

- Başlangıç 212 backend testi; mevcut UI 10 senaryosu eski kanıttır. Frontend/
  canlı sağlayıcı yeniden koşulmaz; upstream validator ortamda yoktur.
- Alert 8 yeni regresyon önce 14 assertion failure/0 error verdi; 48 hedef
  (8 yeni + 11 sinyal + 13 risk + 16 pozisyon) green. Yalnız dört guard ve güçlü
  AL boyut dalı değişti; güçlü SAT eski satış dalını kullanır. OTT okuyucuları aynı.
- DayTrader 10 yeni test önce 43 assertion/subtest failure/0 error; eski 17 ile
  27/27 green. ATR istenen period kadar veri ister; entry sonlu/pozitif fiyat ve
  ATR ister; guard indicator işinden öncedir. Add/close yön hatası kayıt/saat
  değişiminden önce ValueError. Gerçek tarama bozuk sembolden sonra gerçek
  hesaplayıcıyla geçerli BUY döndürür; I/O engelli. Boyut/sermaye politikası eklenmedi.
- OTT 2 test önce 4 pandas TypeError, `.ffill()` sonrası eski 17 teknik testle
  19/19 green. Bu dar runtime testi trend doğruluğu veya yeni sözleşme testi değil.
  Mevcut `[100,110,90]` → `[100,100,100]` fiyatı ve `[NaN,sell,sell]` sinyali
  yeniden üretildi. Seed loop sonrasında; yeni `ott_trend` hâlâ yok. Bu kalan
  yanlış davranış pozitif golden değer olarak testlere konmadı.
- ADR-0006 Proposed: mevcut VAR/period/percent/recurrence korunur; loop önce
  seed, yeni yön alanı ilk NaN/sonra ±1. Bağımsız Fraction örneği VAR
  `[100,107.5,1240/13]`, OTT `[100,105.995,96.72]`. Boş/tek gözlem ve geçersiz
  Close sözleşmesi belgede somuttur. Bu yeni alan/başlangıç henüz kodlanmadı;
  AGENTS veri modeli kapısı gereği açık onay bekler.
- Ana ajan 94 hedef ve birleşik 232 testi dış socket engelli geçti; Windows
  asyncio loopback serbest, create_connection engelli. Bağımsız reviewer 66
  yeni+korunmuş testi ve ADR örneğini kontrol etti; önemli/engelleyici veya
  eyleme dönük minor bulgu yok. Frontend/canlı Yahoo/RSS/SMTP tekrar koşulmadı.
- Diğer riskler: canonical OTT doğruluk/etkinlik araştırması yapılmadı; genel
  dependency audit/destek matrisi, production/ML/M1 kapıları sürer. Yeni endpoint,
  kalıcı veri formatı/migration/dependency yok; commit/push yapılmadı.
- Son yapı kontrolü: 52 Python compile, 39 zorunlu nonempty belge, 11 Boot
  bölümü, yerel Markdown linkleri ve ADR-0004/0005 Accepted / ADR-0006 Proposed
  durumları geçti. Yeni dosya whitespace ve `git diff --check` temiz; config/
  requirements/frontend package/lock aynı. Upstream CDSK validator ortamda yok;
  bu yapısal kontrol onun yerine geçti sayılmaz. HEAD hâlâ `6ed1a59`.
