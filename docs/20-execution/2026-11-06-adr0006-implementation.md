# ADR-0006 OTT uygulama planı

**Goal:** Kullanıcının açıkça onayladığı OTT seed/yön sözleşmesini uygulamak.
**Architecture:** Mevcut TechnicalAnalyzer üreticisi ve AlertSystem okuyucuları;
yeni servis/API/kalıcılık/dependency yok. **Tech Stack:** unittest/pandas/numpy.
**Spec:** Accepted ADR-0006, BACKLOG RP-03.

## Sınırlar

- Kullanıcı somut ADR sorusuna “onaylıyorum” dedi; Accepted kaydedildi. Yalnız
  bu onaylı sözleşme uygulanır. Config 2/1.4, adjusted VAR, max/min recurrence,
  else eşitliği ve tüketici puan ağırlıkları aynı kalır.
- OTT[0]=VAR[0] loop öncesinde; ott_signal ve ott_trend ilk gözlemde eksik,
  sonraki gözlemlerde buy/sell ve +1/-1. İlk yön crossover değildir.
- Boş veride üç boş Series, tek gözlemde fiyat ve eksik yön; sonlu/pozitif
  olmayan Close ValueError, girdi veya önceki sonuç mutasyondan önce reddedilir.
- Kullanıcı girdi DataFrame'i korunur. Desteklenmeyen config/sıralama/diğer
  kolon politikası, canonical OTT/finansal etkinlik ve genel M1 kapsam dışı.
- HEAD `6ed1a59`, aynı dal, önceki değişiklikler korunur; commit/push yok.

## Task 1 — Üretici sözleşmesi (ana ajan)

**Modify:** `modules/technical_analysis.py`, `backend/tests/test_ott_runtime.py`.
**Create:** `backend/tests/test_ott_contract.py`.

- [x] RED: ADR literal tablo, iki yön/eşitlik, ilk/boş/tek gözlem, invalid
  Close ve reddetmede durum korunması; fiyat/index/caller korunması.
- [x] Seed'i loop öncesine taşı, eşlenik ott_trend üret; eksik ilk yön ve
  boş/invalid girdi sözleşmesini uygula. Eski ott/ott_signal isimleri korunur.
- [x] Hedef green ve gerçekçi bellek içi mutantlarla hata yakalama; mevcut
  runtime/hesaplama/risk/pozisyon regresyonları korunur.

## Task 2 — Gerçek okuyucu zinciri (entegrasyon worker)

**Create:** `backend/tests/test_ott_integration.py`.

- [x] Gerçek üreticiden gelen +1→-1 SAT ve -1→+1 AL; ilk/tek yön ve değişmeyen
  yön false crossover üretmez. İç hesaplayıcı/okuyucu mock'lanmaz.
- [x] Gerçek pozisyon okuyucusu son ±1 için mevcut puanı ekler; ilk eksik yön
  puan eklemez (okuyucunun mevcut asgari veri sınırı korunur).
- [x] Caller girdi ve tekrar çağrı korunur; RED/green kanıtı, bilinen sınırlar.

## Task 3 — Birleşim/kapanış

- [x] Tam backend dış ağ engelli (asyncio loopback serbest), bağımsız reviewer,
  Python compile, CDSK 39 belge/11 Boot bölümü, link/ADR/diff/manifest kontrolü.
- [x] DATA_MODEL, OVERVIEW, BACKLOG, TEST_STRATEGY, CHANGELOG ve PROJECT_BOOT
  Accepted/mevcut durumla uzlaşır. RP-03 kapanan kapsam ve kalan genel riskler ayrı.
- [x] Yeni API/UI değişikliği yok; frontend/canlı sağlayıcı tekrar koşulmadıysa
  açıkça yazılır. Yeni modelden kârlılık veya başarı olasılığı iddia edilmez.

## Kanıt

- Başlangıç 232 backend testi, 52 Python dosyası; önceki 10 UI senaryosu kayıtlı.
- Kullanıcı somut ADR-0006 onay sorusuna “onaylıyorum” dedi; karar Accepted,
  eski Proposed aşaması risk-position-fixes kaydında tarihsel olarak korunur.
- İlk 9 contract testi RED: 26 assertion failure ve 5 error; eksik yön alanı,
  guard yokluğu ve empty girdi. 6 gerçek entegrasyon testi RED: 13 eksik-alan
  assertion failure ve 1 empty IndexError. Seed/alan/guard uygulaması sonrası
  önceki runtime 2 testle 17/17 green.
- Kompleks fiyat 3 DataError reproducer'ı reel fiyat guard'ıyla ValueError oldu.
  Reviewer'ın temporal fiyat minor notu ek 1 testte 2 DataError üretti;
  datetime/timedelta guard'ı sonrası 18/18 green. Geçersiz input reddinde caller/
  analyzer/önceki sonuç değişmediği sınanır; eksik ilk yön uydurulmaz.
- Literal varsayılan fiyatlar `[100,110,90]` → `[100,105.995,96.72]`,
  ayna `[100,90,110]` → `[100,93.795,103.15076923076923]`. Özel config
  period3/percent2 için `[100,1568/15,3468/35]`; beklentiler bağımsız matematik.
- 18 geçerli transient mutant assertion failure, error sıfır: seed yok/0,
  adjusted VAR değişimi, config hardcode, max/min ve else eşitliği, legacy
  buy/sell ve trend işaretleri, ilk yön/sinyali uydurma, sıfır/sonsuz fiyat
  kabulü ve stop işaretleri. Gerçek kod geri green; diske mutation yazılmadı.
- Gerçek üretici→crossover/pozisyon zinciri mock olmadan test edilir. İlk veya
  değişmeyen yön crossover sayılmaz; ±1 mevcut OTT puanı 1'i ekler. Empty/single
  yalnız crossover okuyucusunda; mevcut pozisyon okuyucusunun ≥2 bar ihtiyacı
  bu onayla değiştirilmedi. Caller/gapped timezone index ve tekrarlar korunur.
- Bağımsız salt-okunur reviewer önce 17, temporal düzeltme sonrası 18 testi
  çalıştırdı. Important/Critical/eyleme dönük minor bulgu kalmadı. Son birleşik
  248 test dış socket engelli geçti (connect/connect_ex dış hedefler engelli,
  Windows asyncio loopback serbest; create_connection engelli).
- Yeni API/UI/servis/dependency/kalıcı migration yok; frontend/canlı sağlayıcı
  yeniden koşulmadı. Genel BL-001/M1/M2, canonical algoritma/kalibrasyon,
  dependency ve operasyon/gizlilik/dağıtım kapıları ayrı. HEAD `6ed1a59`, commit/
  push yok. Sonraki adım ML yardımcılarının deterministik testleri.
- Son yapı kontrolü: 54 Python compile, 39 zorunlu nonempty belge, 11 Boot
  bölümü, linkler ve ADR-0004/0005/0006 Accepted durumları geçti. Yeni dosya
  whitespace ve `git diff --check` temiz. Config/requirements/frontend package/
  lock değişmedi. Upstream CDSK validator ortamda yok; yapısal kontrol onun
  yerine geçti sayılmaz.
