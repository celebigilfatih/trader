# Momentum indicator tests implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan; steps use checkboxes for tracking.

**Goal:** M1-P0-01'in ilk diliminde gerçek Stochastic, Williams %R ve CCI
hesaplarının sayısal/pencere/okuyucu sözleşmesini ağsız doğrulamak.

**Architecture:** Gerçek `TechnicalAnalyzer.add_indicator` ve
`get_latest_indicators` kullanılır; ta/pandas hesapları mock'lanmaz.
Yalnız sentetik OHLCV ve bağımsız literal/rasyonel beklenen değerler kullanılır.

**Tech Stack:** Mevcut Python/unittest/pandas/ta; yeni bağımlılık yok.

**Spec:** [M1_CLOSEOUT.md](M1_CLOSEOUT.md), M1-P0-01 ilk dilimi;
[PRODUCT_SPEC](../00-product/PRODUCT_SPEC.md) çevrimdışı çekirdek gereksinimi.

## Global constraints

- Mevcut codex çalışma dalında ve checkout'ta devam edilir; kullanıcı dosyaları korunur.
- Formül, warmup, geçersiz girdi, API alanı veya finansal politika değiştirilmez.
  Ortaya çıkan gerçek kusur passing beklentiye çevrilmeden ayrı kaydedilir.
- Default K=14, D=3, Williams=14, CCI=20 yalnız davranış üzerinden doğrulanır.
- Warmup/flat-window NaN geçerli sıfır değerinden ayrılır; kalibrasyon iddiası yok.
- Alt dilim BL-001/M1 bütünü sayılmaz; ADR/runtime/manifest değişikliği hedeflenmez.

## Review focus

- K ve Williams hesabında Close yerine gerçek High/Low ekstremumlarının kullanılması.
- Eski ekstremumun 14 gözlemden sonra çıkması ve D'nin üç geçerli K beklemesi.
- CCI'da HLC typical price, ortalama mutlak sapma ve 0.015 ölçeği.
- Gerçek sıfır/endpoints, yetersiz/flat pencere ve sonradan toparlanma ayrımı.
- Değişen config'in tüketilmesi, index/girdi korunması ve gelecekteki barların prefix'i değiştirmemesi.

## Task 1: Momentum hesapları ve kanıt kaydı

**Files:** Yeni `backend/tests/test_momentum_indicators.py`; execution kaydı,
M1_CLOSEOUT, BACKLOG, TEST_STRATEGY, MILESTONES, OVERVIEW, CHANGELOG ve Boot güncellemesi.

**Interfaces:** Mevcut `TechnicalAnalyzer(data)`, `add_indicator(name)`,
`indicators` serileri ve `get_latest_indicators()`; yeni public arayüz yok.

- [x] Sentetik H/L/C fixture'ı ve dış socket engeliyle gerçek hesaplayıcı testleri yaz.
  Stoch: ilk bar H=200/L=0, sonraki H=120/L=80; C[13:18]=80/100/120/90/110.
  K[13:18]=40/50/100/25/75, D[15:18]=190/3,175/3,200/3.
  Williams[13:18]=-60/-50/0/-75/-25. Sabit 80..120 aralıkta endpoints
  K=0/100/50, Williams=-100/0/-50 ayrıca sınanır.
- [x] Warmup ve reader: K/Williams ilk geçerli pozisyon13, D15, CCI19;
  eksik seri latest çıktısında yok, hesaplanmış sıfır vardır. Boş/tek/kısa
  fixture, flat-denominator NaN ve değişken bar sonrası toparlanma ayrı sınanır.
- [x] CCI: 19 adet TP=10 ardından H/L/C=(20,8,8),(12,4,8) ve TP=10 barları.
  CCI[19]=2000/3, [20]=-2000/3, [21]=0; [39]=2000/57, [40]=NaN.
  Kısa config K/W=3,D=2,CCI=3 için H/L/C=(20,0,10),(12,4,8),
  (14,6,10),(16,8,12),(18,10,14): K[2:]=50,200/3,200/3;
  D[3:]=175/3,200/3; W[4]=-100/3; CCI[2:]=50,100,100.
- [x] Girdi/index ve prefix korunmasını doğrula. Hedef komut:
  `.\.venv\Scripts\python.exe -B -m unittest backend.tests.test_momentum_indicators -v`.
  Expected: bütün hedef testler OK; hesaplayıcı/ta double'ı yok.
- [x] Geçerli bellek içi period/kolon/işaret/pencere/MAD/ölçek/reader mutantlarını
  testlerle assertion failure olarak yakala; error mutantını kanıt sayma.
  Restore sonrası aynı hedef komutu yeniden çalıştır; Expected: OK.
- [x] Tam backend discovery'yi dış socket engelli çalıştır; Expected: mevcut349
  ve yeni testler OK. Python compile/CDSK yapı/link/ADR/diff kontrollerini yap.
  Frontend/canlı smoke bu yalnız test adımında gerekçeli N/A; upstream validator yokluğu ayrı.
- [x] Bağımsız son inceleme, belgeler ve Boot handoff'u güncelle.

**Teslim:** Önceki açık commit/push yetkisiyle aynı dala gönderilir; teslim SHA'sı
Git geçmişi ve uzak ref'ten izlenir. Remote SHA/clean tree kontrolü kapanış
raporunda bildirilir; belgedeki test tamamlanması M1 kapanışı değildir.

## Evidence

- Yeni `test_momentum_indicators.py`: 17/17 gerçek hesaplama testi; ta/pandas
  double'ı yok, dış socket'ler her testte engelli. Open fixture'ı High/Low
  ortasıdır; Close ile aynı yapılıp yanlış kolon kullanımının kaçması önlendi.
- İlk 34 mutant koşusunda 31 assertion kill/3 KeyError vardı. Alan varlığı
  assertion'ları güçlendirildikten sonra **34/34 geçerli mutant assertion
  failure/0 error**; restore sonrası **17/17 OK**. Diskte üretim kaynağı değişmedi.
  Period/config, H/L/C/Open kolonları, D rolling/center/sum, Williams işareti,
  CCI MAD/std/kare/ölçek, warmup filling, zero reader ve index/girdi mutantları.
- Tam discovery **366 test / 28.443 saniye / OK**, dış socket bağlantıları
  engelli, Windows asyncio loopback serbest. 63 Python compile, 39 zorunlu
  CDSK dosyası/11 Boot bölümü, 8 Accepted ADR ve yerel belge bağlantıları geçti.
  Upstream CDSK validator ortamda yok; yapı kontrolü onun başarısı değildir.
- **OS-01 açık:** `analyze_symbol('FIXTURE.IS', period='1y', interval='1d',
  indicators='stoch', email='fixture@example.com')`, 20 sentetik OHLCV barı ve
  yalnız provider boundary değişimiyle `KeyError('stoch')` verdi. Gerçek
  TechnicalAnalyzer stoch_k/stoch_d üretirken route seçici stoch anahtarını okur.
  Williams/CCI aynı koşulda sayısal sonuç verdi. Bu doğrudan route fonksiyonu
  reproducer'ıdır; HTTP/browser veya canlı sağlayıcı testi değildir.
  Hata passing beklenti yapılmadı. Yeni alan/sözleşme seçimi bu test diliminde
  yapılmadı; sonraki öncelik bu sözleşmeyi somutlaştırmak ve gerekiyorsa onay.
- Runtime/API/CI/config/dependency/ADR değişmedi. Frontend build/browser ve
  canlı sağlayıcı bu test adımında yeniden koşulmadı. Genel M1 açık.
- Bağımsız son inceleme: eyleme dönük bulgu yok; 17 hedef test reviewer tarafından
  **0.038 saniye / OK** yeniden çalıştırıldı. Rasyonel oracle'lar ve OS-01 kapsam
  kararı doğrulandı. 366-test/34-mutant sonuçları controller kanıtıdır; reviewer
  bunları, finansal etkinliği veya dış sistemleri bağımsız tekrar doğrulamadı.
