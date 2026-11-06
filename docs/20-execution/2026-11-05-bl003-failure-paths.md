# BL-003 mevcut hata yolları ve timeout iletimi

> Bu kayıt ADR-0005 öncesi dilimin kanıtıdır. Sonraki somut kullanıcı onayı ve
> uygulama `2026-11-05-adr0005-implementation.md` içinde kayıtlıdır; aşağıdaki
> Proposed/mevcut URL parser ifadeleri o önceki aşamayı anlatır.

**Goal:** Mevcut provider/RSS hata yollarını çevrimdışı doğrulamak; provider'ın
tanımlı timeout değerinin gerçek Yahoo history çağrısına gitmesini sağlamak.
**Spec:** BACKLOG sıralı iş 5, API_CONVENTIONS ve mevcut INTEGRATIONS/OVERVIEW.
**Tech Stack:** Mevcut Python/unittest, pandas, yfinance, feedparser.

## Sınırlar

- Mevcut provider varsayılanları korunur: timeout 8, ek retry 2, backoff 1.5.
  Bu değerlerin seçimi değiştirilmez. Legacy çağrılar yeni override verilmezse
  yfinance mevcut varsayılanını kullanır.
- `get_stock_data` sonuna isteğe bağlı keyword-only `timeout_seconds` eklenir;
  repository içindeki çağıranlar incelendi. Enjekte provider fetcher'ı bu
  keyword'ü kabul etmelidir; mevcut test doubles `**kwargs` destekler.
- History timeout bir toplam işlem deadline'ı değildir. Kurulu yfinance 1.8.0
  lazy timezone çağrısında kendi `timeout=10` ayarını kullanır. Bütün HTTP
  yollarının tek bütçeye bağlandığı veya sürüm matrisi doğrulandığı iddia edilmez.
- RSS yeni timeout/retry, hata metriği veya sözleşme alanı bu adımda seçilmez;
  gerekiyorsa ayrı Proposed ADR somutlaştırılır ve onay bekler.
- Önceki kullanıcı değişiklikleri korunur; commit/push, vendor veya dependency
  bildirimi değişikliği yapılmaz.

## Task 1 — Provider / Yahoo adaptörü (ana ajan)

**Files:** `backend/app/providers/market_data.py`, `modules/data_fetcher.py`,
`backend/tests/test_market_data.py`, yeni `backend/tests/test_data_fetcher.py`.

- [x] Kontrollü timeout, başarısız deneme/toparlanma/tükenme, None/boş veri,
  toplam deneme ve 1.5/3.0 backoff sırasını test et; gerçek sleep/ağ kullanma.
- [x] Provider → gerçek BISTDataFetcher → mock `Ticker.history` zincirinde
  timeout iletimini test et; mevcut kodda başarısız olduğunu gör.
- [x] İsteğe bağlı timeout'u ilet; legacy override yokluğu ve finansal OHLCV
  normalizasyonu korunur. History hata halinde None davranışı korunur.

## Task 2 — RSS kısmi kesinti karakterizasyonu (bağımsız test worker)

**Files:** Yeni `backend/tests/test_rss_failure_paths.py`; dar HTTP hata guard'ı
için `modules/sentiment_analyzer.py`. Yeni timeout/retry/tanı politikası uygulanmaz.

Kaynak incelemesinde HTTP hata gövdesinden entry kabulü bulundu. Ana ajan bu
kanıtlanmış hatayı ayrı red→green testle düzeltebilir; yeni RSS timeout/retry,
tanı veya sözleşme politikası seçilmez.

- [x] Kontrollü kaynak timeout/exception, boş HTTP hata yanıtı, kısmi parser
  sonucu ve tüm kaynakların erişilemezliği: kalan gerçek haberler korunur;
  sahte başlık/skor/tarih fallback'i eklenmez.
- [x] Yeni politikayı uygulanmış gibi test etme; mevcut güvensiz/kararsız
  davranış bulunursa sınırlamayı raporla, yanlış davranışı olumlama.

## Task 3 — Kapanış ve karar paketi

- [x] Tam backend paketi dış ağ engelli çalışır; uygun sözdizimi/CDSK yapı ve
  diff kontrolleri geçer. Değişmeyen frontend yeniden build edilmez.
- [x] Bağımsız inceleme; mevcut/hedef ayrımıyla INTEGRATIONS, OVERVIEW,
  OBSERVABILITY, TEST_STRATEGY, BACKLOG, CHANGELOG ve PROJECT_BOOT güncellenir.
- [x] RSS timeout/retry/görünür hata sözleşmesi gerekiyorsa somut Proposed ADR
  hazırlanır; yeni politika veya metrik onaysız uygulanmaz. BL-003/M1 açık kalır.

## Başlangıç / kanıt kaydı

- Dal `codex/m1-pattern-signal-tests`, HEAD `6ed1a59`; 82 test / 10 UI senaryosu
  önceki doğrulama kaydıdır. Çalışma ağacı önceki oturumların değişikliklerini içerir.
- Kaynak kanıtı: provider timeout'u saklıyor, fetcher çağrısına vermiyor.
  Kurulu `PriceHistory.history` timeout parametresini destekliyor. RSS doğrudan
  URL ile `feedparser.parse` kullanıyor; açık request timeout/retry yok.

- Task 1: Geçerli RED 14 testte 1 failure/3 error (timeout zinciri). İletim
  düzeltmesi sonrası 14/14 geçti; varsayılanlar ve legacy call biçimi korundu.
- Task 2: Worker mevcut davranış için 8 ağ/sleep kapalı test ekledi. Kaynak
  incelemesindeki HTTP hata gövdesi kabulü ana ajan tarafından 5 subtest red
  ile doğrulandı; >=400 guard sonrası 10/10 RSS testi geçti. 301 metadata ve
  kullanılabilir bozo entry'ler korundu; yeni retry politikası eklenmedi.
- Task 3: 24 hedef / birleşik 103 test dış socket engelli geçti. 43 Python
  compile kontrolü başarılı. Bağımsız incelemede Important/Critical yok;
  ADR-0005 error_code enum'una Yahoo paragrafıyla uyumlu no_data eklendi.
- ADR-0005 Proposed hazır; RSS bütçe/tanı ve yeni uyarı metni onay bekler.
  Mevcut frontend/live-provider kontrolleri yeniden koşulmadı; dependency
  manifestleri ve önceki kullanıcı değişiklikleri korundu, commit/push yok.
- Son yapı kontrolü: 39 zorunlu CDSK belge, 11 Boot bölümü, ADR durumları ve
  ilgili bağlantılar geçti. Upstream validator Windows ortamında yoktur;
  bu yapısal kontrol onun yerine geçmiş sayılmaz.
