# BL-001 MACD/Bollinger değişken seri test planı

**Goal:** Sabit fiyat testlerinin yakalayamadığı dönem, yön ve yayılım hatalarını
bağımsız beklenen sayılarla yakalamak.
**Architecture:** Gerçek `TechnicalAnalyzer.add_indicator` ve kurulu `ta` çalışır;
beklenen sonuçlar `ta`/pandas hesaplarından veya analiz çıktısından türetilmez.
**Tech Stack:** Mevcut unittest, pandas fixture, standart kütüphane matematiği.
**Spec:** BACKLOG BL-001 bağımsız inceleme açığı; TEST_STRATEGY P0 hesaplama katmanı.

## Sınırlar ve review odağı

- Yalnız testler ve ilgili kayıtlar; üretim formülü, config, dependency, API,
  veri modeli veya ürün kapsamı değişmez. Yeni ADR/onay gerektiren karar yok.
- Başlangıç HEAD `6ed1a59`, dal `codex/m1-pattern-signal-tests`; önceki bekleyen
  değişiklikler korunur. Commit/push bu görevin kapsamında değildir.
- MACD mevcut 12/26/9, recursive EMA (`adjust=False`), ilk Close ile seed;
  çizgi 26., sinyal/histogram 34. gözlemde görünür. Bu davranış test edilir;
  ürün için farklı finansal politika seçilmez.
- Bollinger mevcut 20 gözlem, ±2 popülasyon std (`ddof=0`); pencere dışına çıkan
  aykırı değer etkisini kaybetmeli. 19 gözlemde sonuç uydurulmamalı.
- İki yönde fiyat şoku/reversion, ayrı çizgi/sinyal/histogram, eksik warmup,
  pencere kayması, üst/alt bant ve girdi/index korunması review odağıdır.
- Bu mevcut davranış için test geliştirmesidir: eksik feature RED'i iddia
  edilmez; gerçekçi bellek içi mutantlar assertion failure üretip gerçek kod
  geri geldiğinde green olmalıdır. Üretim dosyası diskte değiştirilmez.

## Task 1 — Gerçek hesaplama testleri

**Modify:** `backend/tests/test_technical_analysis.py`.

- [x] 25 adet 100, bir 126 veya 74, ardından 10 adet 100 fiyat fixture'ı.
  İlk şok MACD `4 - 52/27 = 56/27`; sonraki çizgiler iki geometrik sönümün
  farkı. Sinyal `S[k] = 4/5*S[k-1] + M[k]/5`, `S[0]=M[0]`; bağımsız
  Fraction hesabıyla elde edilen seçili sonuçlar literal tablo olarak sabitlenir.
- [x] Çizgi, sinyal ve histogram için seçili şok/reversion gözlemlerini iki
  yönde sınayan iki test; warmup/sonuç görünürlüğü için bir sınır testi.
- [x] `[1000] + [1..21]` fixture'ında 20 fiyat penceresi için üç bant doğrulanır:
  1..20 ortalama 10.5, popülasyon varyansı 33.25, yarı genişlik `sqrt(133)`.
  Sonraki pencere ortalaması 11.5; önceki aykırı fiyat pencereden çıkar.
- [x] Bollinger 19/20 gözlem sınır testi; mevcut input korunması testine
  Bollinger eklenir, sonuç serilerinin tarih index'i de korunur.

## Task 2 — Hata yakalama ve birleşik doğrulama

- [x] Hedef testler gerçek kodla green; ağ/sleep yok.
- [x] MACD fast/slow/signal dönemleri, çizgi/histogram işareti ve sinyal alanı;
  Bollinger dönem, std çarpanı, bant yönü ve ddof hataları bellek içi değişimde
  ilgili yeni testlerle yakalanır. Her mutant beklenen assertion failure verir;
  kaldırıldığında hedef paket tekrar green.
- [x] Birleşik backend paketi dış socket engelli (Windows asyncio loopback
  serbest). Fresh salt-okunur reviewer yeni testlere ve bu plana bakar.

## Task 3 — Kalıcı kayıt ve kapanış

- [x] BACKLOG, TEST_STRATEGY, OVERVIEW, CHANGELOG ve PROJECT_BOOT handoff/count
  güncellenir. Tarih 2026-11-06; eski kanıtlar tarihsel olarak korunur.
- [x] Python compile, 39 CDSK dosyası/11 Boot bölümü, bağlantı ve whitespace
  kontrolleri. Upstream validator ortamda yoksa açıkça yazılır.
- [x] Canlı sağlayıcı/frontend tekrar koşulmaz; mevcut güvenlik/deprecation,
  Constitution Draft ve genel BL-001/M1 kapanış eksikleri ayrı raporlanır.

## Kanıt kaydı

- Başlangıç: önceki toplam 134 backend testi; frontend 10 senaryosu önceki kanıt.
- Yeni 5 + önceki 12 teknik analiz testi: 17/17 geçti. Beklenen değerler exact
  Fraction/elle varyans hesabından gelir; test sırasında ta/pandas oracle yok.
- Bellek içi mutantlar `patch.dict` (config) ve mevcut metodun transient derlenmiş
  kopyasına `patch.object` ile uygulandı; on-disk üretim kaynağına yazılmadı.
  Her denemede yalnız yeni 5 test koşuldu; aşağıdaki sayıların hepsi assertion
  failure'dır, beklenmeyen error sıfırdır. Patch çıkışında tüm 17 test tekrar geçti.

| Mutant | Yakalanan assertion |
|---|---:|
| MACD fast 12→10 | 22 |
| MACD slow 26→24 | 26 |
| MACD signal 9→5 | 14 |
| MACD çizgi işareti ters | 10 |
| Histogram işareti ters | 6 |
| Sinyal alanına çizgi yazma | 9 |
| Sinyal/histogram erken fillna | 16 |
| Bollinger period 20→19 | 11 |
| Bollinger std 2→1 | 6 |
| Üst/alt bant yer değiştirme | 6 |
| Bollinger erken fillna | 2 |
| Bollinger ddof 0→1 | 6 |

- Birleşik komut: `.venv\Scripts\python.exe -B` ile unittest discover
  (`backend/tests/test_*.py`), socket connect/connect_ex dış hedeflerde engelli,
  `create_connection` engelli. Windows asyncio loopback serbest; 139/139 geçti.
- Bağımsız salt-okunur inceleme exact Fraction ve varyansı ayrıca hesapladı;
  17 hedef testi yeniden çalıştırdı. Critical/Important/Minor bulgu yok.
- API/frontend/üretim hesapları/config ve manifest değişmedi. Yeni ADR yok;
  Constitution Draft, genel M1 ve canlı sağlayıcı/üretim kapıları açık.
- Son kontrol: 44 Python compile, 39 zorunlu nonempty belge, 11 Boot bölümü,
  Markdown bağlantıları/Accepted ADR durumları ve `git diff --check` başarılı.
  Üretim hesap/config ve dependency manifestlerinin HEAD'e göre diff'i boş.
  Upstream CDSK validator Windows ortamında yok; yapısal kontrol onun sonucu değil.
