# ADR-0005 uygulama planı

**Goal:** Onaylanan RSS istek bütçesi, geçici RSS/Yahoo tanıları ve legacy kaynak
uyarısını uygulamak. **Spec:** Accepted ADR-0005; ADR-0004 korunur.
**Tech Stack:** Mevcut Python/unittest, requests, feedparser, Streamlit.
**Architecture:** Mevcut modüller içinde kontrollü transport; tanılar bellek
içinde, yeni HTTP endpoint/kalıcı model/servis/dependency politikası yok.

## Ortak sözleşme ve sınırlar

- Kullanıcı, ADR-0005 onay sorusuna “devam edelim” dedi; karar Accepted kaydedildi.
- RSS connect/read `(8, 8)`, 2 ek retry, 1.5/3.0 bekleme; yalnız ADR'daki
  timeout/connection ve status'lar. 429/503 + Retry-After otomatik tekrarı erteler.
- HTTP gövdesi requests ile alınır, parser yalnız byte görür. Response kapanır;
  parser hatası transport retry tetiklemez. Bozo kullanılabilir entry'yi korur.
- RSS `last_fetch_diagnostics` her fetch'te sıfırlanan kaynak listesi;
  Yahoo aynı adlı alanı her get_ohlcv için tek kaynaklık liste olarak tutar.
  Her kayıt `source/status/attempts/elapsed_seconds/error_code` içerir.
- Özet `source_diagnostics` bağımsız kopyadır; yeni fetch eski özeti değiştiremez.
  NLP hata/sıfır ve yayın/çekilme zamanı ayrımı değişmez.
- Tanılar adapter denemeleridir; SDK içi HTTP sayısı, hard deadline veya global
  sağlık/rate-limit metriği değildir. Kullanıcı/haber/ham exception metni taşımaz.
- Mevcut kullanıcı değişiklikleri korunur; commit/push, migration veya yeni servis yok.

## Task 1 — RSS transport ve testler (RSS worker)

**Files:** `modules/sentiment_analyzer.py`, `backend/tests/test_sentiment_analyzer.py`,
`backend/tests/test_rss_failure_paths.py`, yeni `backend/tests/test_rss_transport.py`.

- [x] Yeni contract için RED: timeout/retry/HTTP/Retry-After, boş/bozo/parse hata,
  kaynak izolasyonu, tanı sıfırlama/kopya ve response cleanup.
- [x] Mevcut 14 ADR-0004 ve 10 RSS hata testinin IO fixture'larını yeni requests
  sınırına uyarla; business assertion'ları ve eski hata regresyonlarını koru.
- [x] Transport/diag ve summary kopyasını uygula; hedef testler ağ/sleep kapalı green.

## Task 2 — Yahoo geçici tanıları (ana ajan)

**Files:** `backend/app/providers/market_data.py`, `backend/tests/test_market_data.py`.

- [x] RED: başarı/retry/tükenme/no_data/timeout/connection/fetch_failed tanıları,
  monotonic elapsed, yeni çağrıda reset ve eski snapshot'ın korunması.
- [x] Mevcut 8/2/1.5 ve dönüş/exception sözleşmesi değişmeden tek kayıt üret.
  Fetcher None verirse neden uydurma; `no_data` yaz. API response alanı ekleme.

## Task 3 — Legacy kaynak görünürlüğü (legacy worker)

**Files:** `app.py`, `backend/tests/test_legacy_disclosure.py`.

- [x] RED: başarısız/kısmi RSS kaynağı varsa onaylı uyarı ve kaynak durumu;
  başarılı/boşta yanlış kesinti uyarısı yok; özet/listenin ayrı fetch tanıları korunur.
- [x] Özetin `source_diagnostics` kopyasını ve listenin kendi son tanısını tüket;
  raw teknik exception gösterme. Eski 8 AppTest davranışı korunur.

## Task 4 — Birleşik kapanış

- [x] Tüm backend testleri dış ağ engelli; 44 Python compile, doc/link/diff
  kontrolleri. Değişmeyen frontend yeniden build edilmez; canlı sağlayıcı ayrı raporlanır.
- [x] Bağımsız diff incelemesi ve gerekli testli düzeltmeler.
- [x] INTEGRATIONS, DATA_MODEL, OBSERVABILITY, TEST_STRATEGY, BACKLOG,
  CHANGELOG ve PROJECT_BOOT güncel. BL-003 yerel kapsamı ile M1/üretim risklerini ayır.

## Başlangıç / ilerleme

- HEAD `6ed1a59`, dal `codex/m1-pattern-signal-tests`; önceki 103 test ve 10 UI
  senaryosu kayıtlı. Önceki staged/untracked değişiklikler korunur.
- Dosya sahipliği ayrıdır: RSS worker modül+RSS testleri, legacy worker UI+AppTest,
  ana ajan Yahoo provider/testleri ve tüm belgeler. Tam paket yalnız birleşimde koşar.

## Uygulama ve doğrulama kanıtı

- RSS: 21 yeni test eski kodda 41 assertion failure + 5 eksik tanı alanı hatası
  üretti. Requests byte parse/policy/tanılar sonrası geçti. Response.close
  istisnasında kaynak tanısı kaybını yakalayan ek 1 test red→green düzeltildi:
  `fetch_failed`, retry yok, sonraki kaynak sürer. Toplam 22 yeni + 24 korunmuş
  test = 46/46. HTTP hata gövdesi, bozo ve tarih/NLP assertion'ları korunur.
- Yahoo: 4 yeni test önce 10 eksik tanı alanı hatası verdi; uygulama sonrası
  provider 13 + adaptör 5 = 18/18 geçti. None/boş veri `no_data`, gözlenen
  timeout/connection/diğer exception sınıfları, toparlanmada hata temizleme,
  deneme/süre ve eski snapshot korunması doğrulandı. Mevcut retry davranışı aynı.
- Legacy: 5 yeni AppTest önce 7 beklenen assertion failure verdi. Özetin
  snapshot'ı ve listenin son fetch tanıları ayrı render edilir; failed/partial
  uyarısı, success/empty yanlış uyarı yokluğu, boş/simge filtreli liste sınandı.
  13/13 geçti; tam monolit/canlı sağlayıcı/browser kanıtı değildir.
- Birleşik: `.venv\Scripts\python.exe -B` ile unittest discover, 134/134 geçti.
  Socket connect/connect_ex dış ağ kapalı, Windows asyncio için loopback serbest;
  create_connection engelli, transport/sleep kontrollü doubles. Canlı Yahoo/RSS/
  SMTP veya frontend build/UI yeniden koşulmadı. Önceki 10 UI senaryosu korunur.
- Bağımsız salt-okunur inceleme: Important/Critical ve eyleme dönük minor bulgu
  yok. Yeni endpoint/kalıcı model/vendor/dependency manifesti eklenmedi; commit/
  push yok. History timeout ve RSS connect/read toplam hard deadline değildir.
- Son yapı kontrolü: 44 Python dosyası compile, 39 zorunlu nonempty belge,
  11 Boot bölümü, Accepted ADR-0004/0005 ve yerel Markdown bağlantıları geçti.
  `git diff --check` ve yeni dosya whitespace kontrolü temiz; requirements ve
  frontend package/lock manifestleri değişmedi. Upstream validator ortamda
  yoktur; bu yapısal kontrol onun yerine geçmiş sayılmaz.
- Açık riskler: SDK yardımcı istekleri, diğer Yahoo metotları, global rate-limit/
  cache, merkezi üretim gözlemi/lisans/operasyon ve genel M1/M2 kapanışı ayrı.
  Next.js kurulum güvenlik uyarısı, Starlette/Streamlit deprecation'ları ve
  upstream CDSK validator ortam eksikliği sürer. Sonraki iş BL-001 değişken
  MACD/Bollinger fixture'larıdır.
