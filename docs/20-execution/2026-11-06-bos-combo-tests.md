# BL-001 BOS ve kombinasyon test planı

**Goal:** BOS, FVG+OB ve FVG+BOS davranışlarını sentetik fiyatlar ve bağımsız
beklenen sonuçlarla doğrulamak; mevcut üretim davranışını değiştirmemek.
**Architecture:** Gerçek AdvancedPatternRecognition ve TechnicalAnalyzer akışları;
test dosyaları ayrıdır, üretim formül/config/API/model/dependency aynı kalır.
**Tech Stack:** Mevcut unittest/pandas ve standart test araçları.
**Spec:** BACKLOG BL-001 ve TEST_STRATEGY P0 deterministik çekirdek kapsamı.

## Ortak sınırlar ve review odağı

- Başlangıç HEAD `6ed1a59`, aynı dal; önceki bekleyen değişiklikler korunur.
  Commit/push veya yeni ADR bu test diliminin kapsamında değildir.
- BOS mevcut geçmiş taraması sondan 10. mumu içerir, daha yeni 9 mumu dışlar; swing seçimi
  çevresindeki gelecek mumları da kullanır. Testler bu batch davranışını
  karakterize eder; canlı/causal strateji, kârlılık veya finansal doğruluk kanıtı değil.
- BOS önceki en yüksek high/en düşük low'u seçer; yorumdaki “en yakın” ifadesi
  bu uygulamayı tam anlatmaz. Bu dilimde fiyat seçim politikası değiştirilmez.
- Kombinasyon confidence alanı mevcut sezgisel formüldür, olasılık değildir.
  BOS strength alanı 100 sınırı uygulandıktan sonraki hacim artışıyla 130 olabilir.
  Sıfır/negatif time_window gibi desteklenmeyen config politikası seçilmez.
- Review: yön simetrisi, eşit/alt/üst eşikler, kapanış yerine wick hatası,
  hacim etkisi, tarih/pencere sınırları, karşıt yön eşleştirme ve fiyat/güç hesabı.
- Test beklentileri üretim çıktısından türetilmez. Yeni testler gerçekçi bellek
  içi mutantlarla başarısız olmalı; disk üretim kaynağına mutation yazılmaz.

## Task 1 — BOS (ana ajan)

**Create:** `backend/tests/test_bos.py`.

- [x] Gerçek OHLCV ile bullish/bearish swing ve kırılma, yayınlanan tarih/fiyat/
  önceki seviye, boş/kısa/flat veri, kapanış eşitliği ve wick-only negatifleri.
- [x] Swing threshold ve hacim/güç hesabı; lookback ve sondan 10. mum sınırı;
  önceki swing'in tarih/ekstremum seçimi, girdi korunması.
- [x] Gerçek TechnicalAnalyzer.add_indicator('bos') sonucu için en az bir
  pozitif örnek ve config'in davranışa etkisi; boş forwarding testi yazılmaz.

## Task 2 — Kombinasyonlar (ayrı test worker)

**Create:** `backend/tests/test_pattern_combos.py`.

- [x] FVG+OB: gerçek uçtan uca pozitifler iki yönde, uyuşmayan yön/mesafe
  negatifleri, mevcut strict %2 fiyat yakınlığı ve çıktı/güç hesabı.
- [x] FVG+BOS: gerçek tespit zinciri pozitifleri ve tarih/fiyat/güç sınırları;
  bağımsız beklenen date/zone/önceki seviye/güç/confidence ve sıralama.
- [x] Tam sınır matcher testlerinde kontrollü gerçek kayıtlar gerekirse ayrı
  açık test sınırı olarak kullanılır; gerçek OHLCV zincirinin yerine geçtiği
  iddia edilmez. Üretim metotlarını mock'layan testler gerekçesiz eklenmez.
- [x] TechnicalAnalyzer kombinasyon çıktısını tüketir; girdi korunur. Yeni
  testlerin hata yakalaması mutantla doğrulanır; kapsam sınırları raporlanır.

## Task 3 — Birleşim, inceleme ve kalıcı kayıt

- [x] Tüm yeni testler ve önceki paket dış ağ engelli (asyncio loopback serbest)
  geçer; bağımsız salt-okunur reviewer yeni testleri/fixture'ları inceler.
- [x] Python compile, CDSK 39 zorunlu belge/11 Boot bölümü, bağlantı/diff ve
  manifest/üretim değişmezliği kontrolleri; upstream validator yokluğu ayrı yazılır.
- [x] BACKLOG, TEST_STRATEGY, OVERVIEW, CHANGELOG, PROJECT_BOOT ve bu kanıt
  kaydı güncellenir. Genel BL-001/M1 ve kalan risk/pozisyon/ML kapsamı açık.

## İlerleme / kanıt

- Başlangıç: 139 backend testi, önceki 10 cockpit senaryosu kayıtlı. Frontend/
  canlı sağlayıcı yeniden koşulmaz; mevcut dependency/deprecation riskleri sürer.
- BOS 15/15 geçti. Fixture 40 mum, index 10'da swing high 125/low 80 ve index
  25'te %1 kapanış kırılmasıdır. Swing gücü 50/40, BOS 30/27; %20 üstü hacimle
  39/35.1. Swing hacmi 2000 için yerel ortalama 1050, oran 40/21; literal BOS
  305/7 veya 265/7. Büyük kırılmada önce 100 sınırı, sonra hacimle 130 mevcut
  sezgisel davranıştır; oran/olasılık değildir.
- Kombolar 12/12 geçti. 26 mumluk OB zinciri %4 hareket/40 güç + gap 1 ile
  20.5 combo; 40 mumluk BOS zincirinde bullish güç 264/7, yakınlık 3/220,
  confidence 37.2; bearish güç 41.2625 ve confidence 40.809065934065934.
  Tüm beklentiler gerçek çıktıdan alınmadan elle hesaplanmış sabitlerdir.
- Matcher testleri ayrı gerçek dedektörlerin FVG/OB cache kayıtlarını kullanır;
  BOS her seferinde gerçek 40-bar veriden hesaplanır, internal metot mock'u yok.
  Tek zincirden gelen iki yönlü pozitifler ayrıca bulunur. Fiyat/güç eşitliği
  komşu float değerleriyle, 3/4 günlük zaman ve strict OB %2 eşiğiyle sınanır.
- BOS 25/25 geçerli transient mutant assertion failure verdi; error sıfır. Kapsanan
  sapmalar: close yerine High/Low, equality, future seviyeler, prior min/max,
  scan/lookback sınırları, break/swing katsayıları, hacim equality/eşik/boost/cap,
  swing eşik/mesafe/komşu/yerel hacim ve analyzer yön alanları. Patch'ler kalkınca 15
  gerçek test tekrar green; diskte üretim kaynağı değişmedi.
- Kombo 15/15 geçerli transient mutant assertion failure verdi; error sıfır: OB %2
  dahil/abs/yön/ortalama, BOS zaman dahil/signed/fiyat/güç/cezalar/sıra/yön/midpoint
  ve analyzer sonuç düşürme. İlk naif yön mutasyonu önceki seviye şemasını bozarak
  KeyError verdi; kanıta dahil edilmedi. Şema-tutarlı yön mutasyonu düzeltildikten
  sonra 15 mutantın tamamı assertion failure üretti; gerçek 12 test yeniden green.
- Birleşik unittest discover `backend/tests/test_*.py`: 166/166 geçti; dış socket
  connect/connect_ex engelli, asyncio loopback serbest, create_connection engelli.
  Gerçek servis/SMTP/sleep yok. Frontend build/UI eski kayıt olarak korundu.
- Bağımsız inceleme 27 hedef testi yeniden çalıştırdı; Important/Critical yok.
  Minor: “en yeni 10 hariç” ifadesi, doğru index sınırına göre “sondan 10. dahil,
  daha yeni 9 hariç” olarak belge/docstring'de düzeltildi; davranış değişmedi.
- Kalan: `get_latest_signals` agregasyonu ve ek yapı/patern kenarları, risk/
  pozisyon/ML, nonpositive time_window ve causal/live kullanım politikası.
  Bu test dilimi genel BL-001/M1/üretim kapanışı değildir; yeni ADR yok.
- Son kontrol: belge/docstring düzeltmesi ardından dış socket kapalı 27/27
  hedef test yeniden geçti. 46 Python dosyası compile, 39 zorunlu nonempty belge,
  11 Boot bölümü, Markdown bağlantı/ADR durumları, yeni dosya whitespace ve
  `git diff --check` temiz. Üretim hesapları/config ve manifestlerin HEAD'e göre
  diff'i boş. CDSK upstream validator ortamda yok; yapısal kontrol onun sonucu değil.
