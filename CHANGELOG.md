# Changelog

## [Unreleased]

### 2026-11-05 — Git teslim hazırlığı

- Kullanıcı commit + push istedi; CDSK belgeleri, kalan iş planı ve mevcut
  API/cockpit/test/geliştirme araçları tek teslim kapsamına alındı.
- README'deki son boşluk düzeltildi; uygulama davranışı ve bağımlılık
  sürümleri değiştirilmedi. Üretilmiş dosyalar ve yerel secrets dışarıda tutulur.
- Teslim öncesi 35 backend testi, CDSK doğrulaması ve UI smoke scripti sözdizimi
  kontrolü ve Next.js üretim derlemesi geçti. Next.js 14.2.18 için kurulum
  güvenlik uyarısı devam ediyor;
  bu bir üretim release'i veya güvenlik riskinin kapatılması değildir.

### 2026-11-05 — Bekleme ve kalan işlerin devri

- Kullanıcı isteğiyle geliştirmeye ara verildi; uygulama kodu değiştirilmedi.
- Execution backlog ve Project Boot handoff'a CDSK durumu, M1–M3 kalan işleri,
  kabul/onay kapıları ve yerel Git remote bilgisi yazıldı.
- Belgeler/yeni uygulama dosyaları için mevcut untracked durum görünür kılındı;
  commit/push yapılmadı ve uzak depo içeriği doğrulanmadı.

### 2026-11-05 — M1 hesaplama doğrulaması

- Gerçek `TechnicalAnalyzer` için 12 deterministik regresyon testi ve gerçek
  hesaplayıcılı 3 API testi eklendi; üretim hesaplamaları/dependencies değişmedi.
- EMA/SMA/VWMA, RSI yönü ve 22 dönemlik ısınma sınırı, karma kazanç/kayıp,
  MACD/Bollinger, destek-direnç, input koruması ve kısa veri uyarısı kapsandı.
- M1 milestone, mimari ve test stratejisindeki eski durum ifadeleri düzeltildi;
  backlog'a kalan beş iş, kabul ölçütleri ve onay sınırları eklendi.
- Doğrulama: 35 backend testi ve CDSK validator geçti. Python 3.9/LibreSSL
  uyarısı devam ediyor; frontend build ve canlı UI smoke bu değişiklik için
  yeniden çalıştırılmadı. M1/M2 tamamlandı veya üretime hazır iddiası yoktur.
- Testlerin ağsızlığı socket bağlantıları engellenerek, EMA/RSI testlerinin
  hata yakalama gücü bellekte yapılan iki kontrollü hesaplama değişikliğiyle
  doğrulandı (üretim dosyaları değiştirilmedi).

### Added

- CDSK 0.1.0 proje iskeleti ve ajan çalışma protokolü.
- Current-state ürün, mimari, execution, kalite, operasyon ve AI belgeleri.
- Önceliklendirilmiş güvenilirlik/ürünleştirme backlog'u.
- ADR-0001 (kapalı beta kapsamı), ADR-0002 (FastAPI/Next.js mimari),
  ADR-0003 (davet + magic link kimlik/bildirim) eklendi.
- `docs/10-architecture`, `docs/20-execution`, `docs/40-operations` ve
  proje özet belgeleri hedef mimari kararı ile güncellendi.
- FastAPI tabanlı ilk backend çekirdeği eklendi (`backend/app`): sağlık,
  davet + magic link auth, sembol ve izleme listesi, teknik analiz, RSI
  tarayıcı ve paper portföy endpointleri.
- Backend servis seviyesinde unit testler (`backend/tests/*`) eklendi (auth,
  watchlist, portfolio, response contract) ve API başlatma scripti
  (`run_api.py`) eklendi.
- Next.js MVP cockpit’u eklendi (`frontend/app/*`): demo auth, watchlist,
  teknik analiz, RSI tarama ve portföy önizleme akışlarını gösteren tek sayfa
  kullanıcı arayüzü.
- `frontend/package.json`, `frontend/package-lock.json` ve Next.js build desteği
  eklendi; `.next` üretim artifaktının repoya yazılmaması için `.gitignore`
  güncellendi.
- Tek komutla geliştirme başlatma scripti eklendi (`run_dev.py`) ile API ve
  Frontend aynı süreçte başlatılabiliyor; bağımlılık kontrolü ve health wait
  mantığı eklendi.
- `run_dev.py` için Windows çalıştırıcısı eklendi (`run_dev.bat`) ve frontend
  bağımlılığı eksik olduğunda isteğe bağlı otomatik `npm install` seçeneği eklendi.
- `run_dev.py` başlatma/sağlık kontrolü ve kapanış davranışı güçlendirildi;
  `--ready-timeout`, `--check-interval`, `--shutdown-timeout` seçenekleri eklendi.
- `scripts/run_dev_smoke.py` eklendi; `run_dev.py` için kısa süreli entegrasyon smoke
  kontrolü (hazır olma ve kontrollü kapanış) genişletildi; auth invite/magic,
  watchlist ve portfolio kontrat doğrulamaları da eklendi.
- `scripts/run_dev_smoke.py` ayrıca front-end kök sayfası HTML marker kontrolü ile
  cockpit doğrulamasını da kapsayacak şekilde genişletildi.
- Frontend smoke kontrolü, Next.js dev-mode chunk URL çıkarımı için esnek regex ve yedek
  `/_next/static/chunks/app/page.js` fallback ekleyerek, bölüm metinlerinin birden
  çok adımda doğrulanmasını sağlayacak şekilde sağlamlaştırıldı.
- Smoke test kapsamı workflow seviyesine taşındı; API tarafında watchlist için
  add/remove/set ve portfolio için add/re-add/remove akışları doğrulanarak temel
  uçtan-ucuna kritik kullanıcı iş akışları smoke’e dahil edildi.
- Tarayıcı tabanlı opsiyonel UI smoke senaryosu eklendi (`scripts/run_frontend_ui_smoke.mjs`).
- UI smoke senaryosunun kırılgan noktasını azaltmak için `frontend/app/page.js`
  tarafında test idleri (`data-testid`) eklendi.
- `scripts/run_frontend_ui_smoke_with_stack.py` eklendi; `run_dev.py` ile backend ve
  frontend’i birlikte başlatıp durduran ve Playwright UI smoke testini otomatik
  çalışan tek adımlı akışı sağlar.
- RSI screener API yanıtına `skipped_symbols` alanı eklendi; çekilemeyen semboller
  ayrı bir alanla dönüyor ve frontend’de kullanıcıya görünür uyarı olarak sunuluyor.
- `skipped_symbols` ile birlikte kontrat `warnings` alanını da getirir oldu; her
  atlanan sembole `symbol`, stabil `code` ve `reason` eklenerek nedeni gösterilir.
- RSI screener’de `warnings` etiketleri kullanıcı dostu görünüme kavuşturuldu;
  teknik `code` değerleri hala korunurken ekran metninde anlamlı Türkçe etiketler
  (`Veri alınamadı`, `RSI hesaplanamadı`) gösterilir.
- RSI `warnings` kartlarında görsel iyileştirme yapıldı; uyarılar ikonlu ve seviye
  rengiyle ayrıştırılarak listeleniyor (kritik/öneri ayrımı için temel tonlama).
- RSI uyarı paneline özet+detay akışı eklendi: atlanan toplam sembol sayısı,
  kod bazlı sayım etiketleri ve istemci tarafında detay aç/kapa kontrolü.
- Uyarı özet etiketleri, kullanım yoğunluğuna göre (sayıya göre) sıralanacak şekilde
  güncellendi; bu da aynı anda birden fazla kod türü geldiğinde okunabilirliği artırıyor.
- `run_dev.py` artık frontend başlatılırken `NEXT_PUBLIC_API_BASE` enjekte ediyor;
  frontend `page.js` bu değişkeni varsa onu öncelikli olarak kullanacak şekilde
  güncellendi.
- MIT lisans referansını destekleyen `LICENSE` dosyası eklendi.
- RSI screener ekranındaki sonuç tablosu (`frontend/app/page.js`) artık her satırda
  veri kaynağı (`source`), çekilme zamanı (`fetched_at`) ve yaşlanma bilgisi
  (`staleness_seconds`) göstermeye başlandı.
- Makefile eklendi; `make run-dev`, `make test`, `make smoke`, `make ci` ile
  tekrarlanabilir geliştirme ve doğrulama akışı oluşturuldu.
- GitHub Actions CI eklendi (`.github/workflows/ci.yml`) ve `push`/`pull_request`
  tetiklemeleriyle otomatik `make ci` akışı çalıştırılıyor.
- `backend/app` içindeki yanlış paket göreceli importları düzeltildi; bu sayede
  backend ve frontend birlikte smoke testleri başarılı oldu.
- `MarketDataProvider` test odaklı olarak fetcher enjeksiyonunu destekleyecek şekilde
  genişletildi ve bu kapsamda `backend/tests/test_market_data.py` eklendi.
- Offline deterministik API kontrat testleri eklendi: `backend/tests/test_api_contracts.py`.
- Aynı kontrat paketi içinde auth ve giriş doğrulama (401/400/422) senaryoları
  eklendi; meta alanları (`source`, `fetched_at`, `as_of`, `staleness_seconds`)
  doğrulaması genişletildi.
- RSI screener cevaplarına da veri kaynağı/uyarı meta alanları eklendi
  (`source`, `fetched_at`, `as_of`, `staleness_seconds`), böylece tarama sonuçları
  da izlenebilir hale geldi.

### Changed

- README'ye CDSK başlangıç yolu, prototip olgunluk notu ve venv tabanlı yerel
  başlangıç eklendi.
- Ignore politikası yerel venv, test/lint cache, coverage ve Streamlit secret
  dosyalarını kapsayacak şekilde genişletildi.
- ROADMAP, MILESTONES ve deployment/ops dokümantasyonu kapalı beta hedefi ve
  Hostinger/Coolify geçişiyle uyumlu hale getirildi.
- Dokümantasyonlarda Frontend Cockpit başlatma adımları ve backend+frontend
  paralel çalıştırma talimatları güncellendi.
- `CHANGELOG.md` ve `PROJECT_BOOT.md` güncellendi; `make ci` validasyonunun
  backend testleri + frontend build + smoke geçişini de kapsayacak şekilde kaydı
  güncellendi.
- Frontend smoke kontrolü, geliştirme orkestrasyonunda kullanıcı arayüzü ayağa
  kalkışını da doğrulayacak şekilde güncellendi.
- UI smoke senaryosu, `frontend/app/page.js` test id’lerini kullanacak şekilde
  daha kararlı hâle getirildi.
- `scripts/run_frontend_ui_smoke.mjs`, frontend bağımlılığına fallback yaparak
  Playwright modülünü `frontend/node_modules/playwright` konumundan da yükleyebilir.
- `make smoke-ui-stack` hedefi eklendi; stack başlatarak UI e2e smoke testini
  tek komutla çalıştırır.
- `make smoke-all` hedefi eklendi; `make smoke` ve `make smoke-ui-stack` ardışık
  çalıştırarak tek komutla uçtan-uca kapsamlı doğrulama sağlar.
- `make ci-e2e` hedefi eklendi (`make ci-core` + `make smoke-ui-stack`) ve geliştirmeye odaklı
  e2e kapsamını CI akışına bağladı.
- `make ci-full` hedefi eklendi (`make ci-e2e`) ve tam otomatik doğrulama zincirini
  tek komutta topladı.
- Yerel doğrulama playbook’u (`smoke`, `smoke-all`, `ci-e2e`, `ci-full`) netleştirildi.
- `make ci-core` hedefi eklendi; `make ci` akışını (deps/test/build+smoke) ve
  `make ci-e2e` akışını (`ci-core` + `smoke-ui-stack`) birbirinden bağımsız ve
  port çakışmasız hale getirmek için.
- `README` hızlı görevler listesine `make ci-core` eklendi; `ci-e2e`/`ci-full`
  akışının neyi içerdiği daha net hale getirildi.
- `make qa` hedefi eklendi; `ci-e2e` için günlük kapanışta okunabilir tek adımlık
  alias sağlandı.

### Validation

- CDSK validator geçti.
- 15 Python kaynak dosyasının sözdizimi kontrolü geçti.
- API katmanı (`backend/app/*.py`) için `py_compile` doğrulaması başarılı.
- Next.js uç kısmı için `npm install` ve `npm run build` ile derleme başarılı.
- Backend smoke testleri `python3 -m unittest discover -s backend/tests -p 'test_*.py'`
  ile **20 test** başarılı geçti.
- `run_dev.py --help` yeni opsiyonlarla birlikte doğrulandı; scriptin hata ve erken
  kapanış senaryoları manuel olarak test edildi.
- `python3 scripts/run_dev_smoke.py` ile kısa bir entegrasyon smoke koşusu doğrulandı
  (auth + watchlist add/remove/set ve portfolio upsert/re-upsert/delete workflow’ları +
  frontend HTML marker doğrulaması).
- `make smoke-ui` hedefiyle Playwright kurulu ise tarayıcı üzerinden temel UI etkileşim
  senaryoları çalıştırılabilir (opsiyonel).
- `make smoke-ui` senaryosu gerçek backend+frontend ortamında çalıştırılarak
  otomatik olarak doğrulandı.
- `make smoke-ui` hedefi çağrıldığında Playwright yüklü değilse açık kurulum
  yönergesi ile güvenli erken çıkış yapılır.
- `scripts/run_frontend_ui_smoke.mjs` için sözdizimi kontrolü (`node --check`) ve
  `make smoke-ui` davranışı doğrulandı.
- `scripts/run_frontend_ui_smoke_with_stack.py` için sözdizimi kontrolü
  (`python3 -m py_compile`) ve yeni `make smoke-ui-stack` hedefi doğrulandı.
- `scripts/run_frontend_ui_smoke.mjs` RSI tarama akışını da doğrulayacak şekilde
  güncellendi (`rsi-screen-button`, `rsi-results-table`) ve tablo başlıklarını kontrol
  edecek ek doğrulama eklendi.
- `scripts/run_frontend_ui_smoke.mjs` uyarı paneli varsa detay açma akışını da test eder
  hale geldi; panel detayları için selector tabanlı test kimlikleri eklendi.
- `scripts/run_frontend_ui_smoke.mjs`, RSI uyarı paneli detay görünürlüğünün
  kalıcılığını doğruluyor: paneli açtıktan sonra localStorage anahtarı
  (`trader_rsi_warning_details_visible`) beklenen değere ayarlanıyor mu kontrol
  ediliyor; yenileme sonrası panelin açık durumda kalması da doğrulanıyor.
- RSI uyarı paneli varsa, “Detayları göster” etkileşimiyle detay listesi renderı
  da Playwright UI smoke tarafında doğrulanabilir oldu (panel yoksa esnek davranış korunuyor).
- RSI screener endpointi, tek sembolde veri yoksa 500 döndürmek yerine o sembolü
  atlayıp kalan sonuçları döndürecek şekilde dayanıklılığı artırıldı.
- `make smoke-all` hedefi doğrulandı (`make smoke` + `make smoke-ui-stack` ardışık
  çalıştırması).
- `make ci` ile tam otomatik doğrulama koşusu başarıyla geçti (backend testler,
  frontend build ve run_dev smoke).
- `make ci-e2e` doğrulandı (`make ci-core` + `make smoke-ui-stack` ardışık koşusu).
- `make ci-full` doğrulandı (`make ci-e2e` ardışık koşusu).
- `make qa` doğrulandı (`make ci-e2e`).
- `backend/tests/test_api_contracts.py` içinde RSI screener sonuçları için kaynak
  ve zaman damgası meta alanlarının kontrat doğrulaması eklendi; test sayısı ve
  kalite kapsamı korundu.
- `backend/tests/test_api_contracts.py` içinde RSI için `insufficient_ohlcv_for_rsi`
  uyarı kodunu doğrulayan senaryo eklendi; böylece yetersiz teknik gösterge veri
  akışı da kontrat seviyesinde kapsandı.
- RSI uyarı panelindeki “Detayları göster/gizle” tercihi artık kalıcı hale
  getirildi; kullanıcı tercihi `localStorage` içinde saklanıp sonraki taramalarda
  korunur.
- Dokümantasyon tutarlılığı: `README`'deki lisans referansı artık gerçek
  `LICENSE` dosyasıyla eşleşiyor.
- Frontend `npm run build` doğrulandı; RSI screener tablosu, API’den gelen meta
  alanları render edecek biçimde render katmanıyla uyumlu hale getirildi.
- RSI screener kontratında `skipped_symbols`/`warnings` alanları ve frontend uyarısı
  da Playwright UI smoke senaryosuna eklendi; `warnings` artık standart kodlarla
  (`market_data_unavailable`, `insufficient_ohlcv_for_rsi`) dönüyor.
- RSI uyarı paneli için detay render elemanları `data-testid` ile etiketlendi ve
  test davranışını kırılgan selector aramalarına değil, kontrol edilen etkileşime
  dayandırarak stabilize edildi.
- RSI uyarı paneli, aynı akışta hem özet hem detay sunarak kullanıcıya
  daha odaklı bir tarama deneyimi sunacak şekilde güncellendi.
