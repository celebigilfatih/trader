# Test Strategy

## Risk-based Test Layers

1. **P0 finansal hesaplama birimleri:** İndikatör, sinyal, patern, feature ve risk
   hesapları deterministik OHLCV fixture'larıyla ağsız test edilir.
2. **P0 veri sözleşmesi:** Yahoo yanıtı normalize edilir; eksik/boş/yetersiz veri
   senaryoları kontrollü sonuç verir.
3. **P1 entegrasyon:** Yahoo/RSS/SMTP başarı, timeout, rate limit ve hata
   davranışları mock'lanır; az sayıda canlı smoke test ayrı çalışır.
4. **P1 UI smoke:** Altı ana navigasyon ekranının açılması ve kritik hata
  göstermemesi hedeflenir; `scripts/run_dev_smoke.py` içinde Next.js kök sayfası
  için HTML marker doğrulaması (başlık, Genel, Giriş) ve app/page chunk metninden
  bölüm ipuçları kontrolü (Watchlist, Teknik analiz, Paper portföy, RSI tarama)
  yanında, kritik iş akışlarının API düzeyinde uçtan-ucuna kontrat teyidi yapılır:
  watchlist ekleme/silme/set ve paper ekleme/tekrar ekleme/silme.
5. **Opsiyonel, canlı veriye bağlı UI e2e smoke:** `scripts/run_frontend_ui_smoke.mjs` ile,
   oturum token’ı ile giriş sonrası watchlist ekleme ve portföy ekleme/silme,
   temel UI etkileşim adımları doğrulanır.
   RSI uyarı paneli kontrolleri panel varsa çalışır; bu nedenle her koşuda
   uyarı/tercih senaryolarının sınandığı iddia edilmez. Deterministik tarayıcı
   senaryoları backlog'da ayrı eksiktir.
6. **P2 çıktı:** PDF ve opsiyonel CSV/model artifact'ları içerik ve güvenli dosya
   davranışıyla doğrulanır.

## API and workflow gates (hedef mimariye geçiş)

- Her `/api/v1` endpoint için en az bir sözleşme testi (`status`, `error_code`,
  `correlation_id`) planlanır.
- Gün sonu snapshot iş akışında en az bir idempotency ve `staleness` testi eklenir.

## Commands

- **Current smoke:** `PYTHONDONTWRITEBYTECODE=1 python3 -B test_app.py`
  (Yahoo Finance ağına bağlıdır; `beautifulsoup4` import adı nedeniyle mevcut
  haliyle yanlış negatif üretebilir).
- **Unit / offline API:** `make test-backend` veya
  `python3 -m unittest discover -s backend/tests -p 'test_*.py'`.
  2026-11-05 doğrulaması: 35 test. `test_technical_analysis.py` gerçek hesaplayıcıyı
  sabit sentetik OHLCV ile sınar; `test_api_contracts.py` hem sözleşme doubles'ını
  hem gerçek hesaplayıcıyla analiz/RSI akışlarını kapsar. Beklenen değerler
  hesaplayıcının kendi çıktısından türetilmez. RSI mevcut konfigürasyonda 22 dönemdir.
- **Integration:** Yukarıdaki çevrimdışı API testleri provider yerine fixture
  kullanır. `make smoke` yerel stack/HTTP entegrasyonudur; canlı sağlayıcı
  doğrulamasıyla çevrimdışı birim test aynı kalite kanıtı sayılmaz.
- **End-to-end:** `scripts/run_dev_smoke.py` içinde temel UI kontratlarının yanında,
  kritik veri akışları için küçük kapsamlı uçtan-ucuna API senaryoları da eklenmiştir
  (watchlist ve paper ekleme/silme döngüleri). İsteğe bağlı olarak, `make smoke-ui`
  ile tarayıcı üzerinden temel UI e2e smoke senaryosu çalıştırılır.
- **Security:** Secret taraması ve dependency audit henüz yapılandırılmadı.
- **Syntax/build minimum:** Repository içindeki her `.py` dosyasını `compile()`
  ile kontrol et; Streamlit için ayrı build adımı N/A.
  Next.js için `make build-frontend` vardır; yalnızca backend test/belge
  değişikliğinde frontend build yeniden koşulmadıysa açıkça belirtilir.
- **CDSK:** `python3 /Users/fatih/Dev/cdsk/scripts/validate_project.py .`
- **Transition:** Backend API contract, cron schedule ve health endpoint için ek gate.

## Test Data and Isolation

Unit testler sabit seed ve küçük sentetik OHLCV fixture'ları kullanmalı; gerçek
ağ, saat ve global Streamlit state birim testlerden ayrılmalıdır. Gerçek finansal
kullanıcı verisi fixture yapılmamalıdır.

## Release Gates

- P0 test başarısızsa release olmaz.
- Ağ yokluğu birim testleri engellemez; canlı smoke test “skipped/degraded” olarak
  ayrı raporlanır.
- Mock/gerçek veri karışması ve sessiz entegrasyon hatası release blocker'dır.
- CDSK validator, sözdizimi ve dokümantasyon/changelog kontrolü zorunludur.
