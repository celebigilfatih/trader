# Test Strategy

## FVG/OB grafik ve ilk legacy RSI — 2026-11-06

11 yeni gerçek helper+actual Plotly chart testi,6 gerçek legacy RSI ve2 actual
buton AppTest. Grafik RED6 assertion failure/0error →GREEN; yalnız DataFrame
okuyucu bağlandı. RSI geçerli mevcut davranış testleri ilk koşuda8/8; hesaplayıcı
double yok, yalnız provider sınırı sentetik OHLCV/None/exception ile kontrol edilir.
Hedef45/45 (3.694 s),14 geçerli bellek içi mutant assertion failure/0error;
tam431/431 (18.175 s) dış socket engelli, asyncio loopback serbest geçti;
ilk sandbox temp-dizin/indeksleme harness koşuları bu sayıya dahil değildir.
FVG/OB iki yön/threshold/lookback/zone/date/filled/strength, seçili/kapanmış grafik,
5/10 bar width/end clamp; RSI65.625/0/100/35/65,bounds,sort,interval ve ilk hazır
bar/no-match/tek hata sonrası gerçek sonuç doğrulandı. Tüm app/browser testi değildir.
API500 ve kesinti=no-match tanıları **beklenen passing test değildir**; OS-02/
RS-01 için ADR-0011/0012 Proposed. [Uygulama/kanıt](../20-execution/2026-11-06-fvg-ob-legacy-rsi.md).

## VST düzeltmesi — Accepted ADR-0010, 2026-11-06

Yeni19 core testi RED114 assertion failure/0 error; üç gerçek API regresyonu
RED3 failure/0 error; dört kontrollü kart testi RED7 failure/0 error verdi.
Üretim düzeltmesi sonrası hedef53/53; tam412/412 (32.881 s), dış socket engelli,
Windows asyncio loopback serbest. Rasyonel hacim/prefix/zero-gap/leading/all-zero/p1,
underflow/finite extremes, strict config/domain ve atomik hata, Supertrend0..p+1,
gerçek risk/sinyal ve HTTP null/weighted sonuçları sınandı. Kart actual assignment
ve with AST'si ile AppTest'te çalışır; tam uygulama/browser kanıtı değildir.
15 geçerli bellek içi mutant assertion failure/0 error ile yakalandı; ilk harness
anchor hatası mutant sonucu sayılmadı, diskte üretim mutasyonu yapılmadı.
[Uygulama/kanıt](../20-execution/2026-11-06-adr0010-implementation.md).
VST-01–03 dar dilimi kapandı; canonical/finansal etkinlik ve genel M1 açıktır.
Frontend/canlı/uzak CI tekrar koşulmadı; upstream CDSK validator ortamda yoktur.

## Önceki VWEMA/Supertrend geçerli alt kapsam — 2026-11-06

Yeni12 core ve1 gerçek API testi; hedef27/27 (0.289 s), tam386 test (28.692 s)
dış socket engelli geçti.32 geçerli transient mutant assertion failure/0 error;
restore27/27. Ta/pandas ve analyzer gerçek; yalnız manuel ATR fallback'i için
dependency exception enjekte edilir. Sabit seed sonrası VWEMA5/20 alpha,
sıfır hacim taşıma/config, gerçek ATR/gap/HL2 ve ayrı rolling fallback,
yön/eşitlik/warmup, index/girdi; gerçek public generate_signal ve API scalar
okuyucuları literal/rasyonel oracles ile sınanır.

O aşamada VST-01/02/03 yeniden üretilmiş açık davranışlardı; passing test yapılmadı.
VWEMA prefix testi yalnız aynı sabit başlangıç penceresinden sonraki geçmişi
kapsar; genel causality iddiası değildir. Supertrend mevcut temel bant/yön
hesabıdır, canonical trailing standart veya finansal etkinlik doğrulanmaz.
Doğrudan9 barla pozisyon okuyucusu tanısı, fetcher'ın mevcut15 bar alt sınırını
değiştirmez veya canlı UI olayı kanıtlamaz. Üretim/CI/dependency/ADR aynı;
frontend/browser/canlı/uzak CI tekrar yok, upstream validator yokluğu sürer.
[Plan/kanıt](../20-execution/2026-11-06-vwema-supertrend-tests.md).

## Accepted ADR-0009 uygulama doğrulaması — 2026-11-06

`test_stochastic_api.py`7 gerçek HTTP regresyonu ekledi: analyzer/ta ve auth
gerçek, yalnız provider/state sınırı kontrollü. %K75≠%D200/3, geçerli0,13-bar
warmup ve20-bar flat-null; mixed seçimde diğer scalar/null, fiyat/meta/correlation,
query400 ve oturum401. Dış socket'ler engelli, asyncio loopback serbest.
Önce5 assertion failure/0 error (HTTP500≠200), dar API alias'ından sonra7/7.
Mevcut API/momentum ile **38/38 /0.337 s /OK**; tam paket
**373 test /27.902 s /OK**. 10 geçerli bellek içi alias/D/bar/alan/null-zero
mutantı assertion failure/0 error; restore7/7. İlk iki geçici harness başlatma/
restore hatası mutant başarısı sayılmadı; diskte üretim mutasyonu yok.
64 Python compile/39 CDSK dosyası/11 Boot/9 Accepted ADR kontrolü geçti.
Frontend build/browser, canlı sağlayıcı ve uzak CI yeniden koşulmadı;
upstream validator yokluğu sürer. Genel M1/finansal etkinlik kapanışı değildir.
[Uygulama kaydı](../20-execution/2026-11-06-adr0009-implementation.md).

## Önceki OS-01 sözleşme incelemesi — 2026-11-06

Üretim/test dosyaları değişmeden gerçek FastAPI TestClient + fixture provider
ile `stoch`500, Williams200(-25), CCI200(null warmup), EMA200; `stoch_k/stoch_d`
400 ve kimliksiz401 gözlendi. Gerçek analyzer K=75/D=200/3 verdi. Mevcut14 API
ve17 momentum testi **31/31 /0.237 s /OK**, dış socket engelli (loopback serbest).
Bu koşu düzeltme kanıtı değildir;500 normatif passing test yapılmadı. Önerilen
200/%K, zero/null/mixed/meta/400/401 regresyonları onay sonrası yazılacaktır.
[ADR-0009](../50-decisions/ADR-0009.md) o aşamada Proposed idi;
[tasarım/kabul kaydı](../20-execution/2026-11-06-stochastic-api-design.md).
Tam366-test kaydı önceki dilime aittir; bu belge adımında frontend/canlı/uzak CI
ve upstream CDSK tekrar doğrulanmadı. Runtime/manifest/CI değişikliği yoktur.

## Momentum hesaplama dilimi — 2026-11-06

`test_momentum_indicators.py` 17 gerçek hesaplama testi ekler: K14/D3 ve
Williams14 için ayrı H/L/C, ekstremumun pencereden çıkması, ilk geçerli değerler,
endpoints/sıfır; CCI20 için typical price/MAD/0.015 ve iki impulse'un çıkışı.
Config override, flat sonrası toparlanma, girdi/index/prefix de sınanır.
Ta/pandas mock'lanmaz; literal/rasyonel beklenen değerler bağımsızdır.
34 geçerli transient mutant assertion failure/0 error; restore sonrası 17/17.
İlk koşudaki üç KeyError mutant sonucu alan-varlığı assertion'larıyla güçlendirildi;
error sonuçları mutation başarısı sayılmadı. Tam paket **366 test / 28.443 s / OK**,
dış socket engelli, Windows asyncio loopback serbest. Üretim kodu değişmedi.

OS-01: gerçek analiz route fonksiyonu fixture provider ile `stoch` seçimi için
`KeyError('stoch')` verir; Williams/CCI geçer. Hatalı davranış passing beklentiye
çevrilmedi. Bu çekirdek dilimi API çoklu çıktı okuyucusunu veya bütün UI'yı
doğrulamaz. Frontend build/browser ve canlı sağlayıcı tekrar koşulmadı.
Plan/kanıt [execution kaydındadır](../20-execution/2026-11-06-momentum-indicator-tests.md).

## Önceki M1 kapanış kontrolü — 2026-11-06, 349 test

[M1 kapanış kontrolü](../20-execution/M1_CLOSEOUT.md) mevcut kabul ölçütlerini
test ve aktif okuyucu envanterine bağlar. Yeni tam koşu dış socket engelli
349 test / 53.874 saniye / OK; Windows asyncio loopback serbesttir. Üç gerçek
modüldeki çağrı profili Stochastic/Williams %R/CCI ve legacy `StockScreener`
yollarında sıfır çağrı gösterdi. Çağrı envanteri satır/branch coverage veya
aritmetik doğruluk kanıtı değildir; FVG/OB alt sınıflarının mevcut testlerini
yok saymaz. M1 ve REQ-007 genel kapanışı kısmi kalır.

Makefile test hedefi genel dış ağ engeli uygulamaz; bu koşunun guard'ı geçici
çalıştırıcıdaydı. Kalıcı offline çalıştırıcı ve CI bağlantısı ayrı kalite işidir.
Mevcut workflow PR/`main` push üzerinde yalnız `make ci` çalıştırır;
`ci-e2e`/`ci-full` veya 10 deterministik browser senaryosu otomatik koşulmaz.
Bu belge adımında frontend build/browser ve canlı sağlayıcılar tekrar koşulmadı.
Önceki doğrulama bölümleri tarihsel dilim kanıtları olarak korunur.

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
5. **Deterministik tarayıcı:** `scripts/run_frontend_ui_smoke.mjs --deterministic`
   ve `scripts/frontend_ui_scenarios.mjs` gerçek cockpit'i sabit route yanıtlarıyla
   sınar. On bağımsız context; auth/watchlist/analiz/paper/RSI başarı, eski
   sonuçtan boş sonuca geçiş, kısmi veri ve açık/kapalı RSI tercih kalıcılığı,
   tüm sembollerde veri yokluğu, 503 sonrası temizleme ve yeniden deneme.
   API istek gövdeleri/auth başlığı kontrol edilir; beklenmeyen API/dış istek
   veya pageerror koşuyu başarısız kılar. Eksik Playwright başarı sayılmaz.
   Analiz/RSI kaynak türleri, eksik/bozuk metadata, hata/sembol değişiminde
   temizleme ve geç gelen eski/önceki oturum yanıtları da sınanır.
6. **Opsiyonel, canlı veriye bağlı UI e2e smoke:** `scripts/run_frontend_ui_smoke.mjs` ile,
   oturum token’ı ile giriş sonrası watchlist ekleme ve portföy ekleme/silme,
   temel UI etkileşim adımları doğrulanır.
   RSI uyarı paneli kontrolleri panel varsa çalışır; bu nedenle her koşuda
   uyarı/tercih senaryolarının sınandığı iddia edilmez. Bu davranış yukarıdaki
   deterministik modda her koşuda ayrıca sınanır; canlı sağlayıcı doğrulaması değildir.
7. **P2 çıktı:** PDF ve opsiyonel CSV/model artifact'ları içerik ve güvenli dosya
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
  2026-11-06 güncel doğrulaması: 386 test. `test_technical_analysis.py` 17 testle gerçek hesaplayıcıyı
  sabit sentetik OHLCV ile sınar; `test_api_contracts.py` hem sözleşme doubles'ını
  hem gerçek hesaplayıcıyla analiz/RSI akışlarını kapsar. Beklenen değerler
  hesaplayıcının kendi çıktısından türetilmez. RSI mevcut konfigürasyonda 22 dönemdir.
  `test_pattern_recognition.py` (14 test) mum, çift tepe/dip, FVG eşik/dolum ve
  order block yön/güç örneklerini; `test_alert_signals.py` (11 test) eşik/kesişim,
  trend, Bollinger/hacim ve birleşik sinyali sınar. Bu kapsam paternin finansal
  geçerliliğini veya tüm patern/yapı/risk/ML yardımcılarını doğrulamaz.
  `test_bos.py` (15 test) ve `test_pattern_combos.py` (12 test) temel BOS,
  gerçek FVG+OB/FVG+BOS zincirleri ve ayrı matcher sınırlarını kapsar.
  `test_risk_analysis.py` (13), `test_position_recommendation.py` (16) ve
  `test_day_trader_risk.py` (17) desteklenen risk/pozisyon/ATR/P&L dilimini kapsar;
  Sonraki `test_risk_position_fixes.py` (8), `test_day_trader_validation.py`
  (10) ve `test_ott_runtime.py` (2) bilinen hataların düzeltme regresyonlarıdır.
  Accepted ADR-0006 için `test_ott_contract.py` (10) ve `test_ott_integration.py`
  (6) fiyat/yön/girdi sözleşmesini ve gerçek okuyucuları kapsar. Bu, bütün
  risk/pozisyon entegrasyonunun veya finansal etkinliğin doğrulandığı anlamına gelmez.
  Eklenen `test_sentiment_analyzer.py` 14 RSS/NLP sözleşme testi;
  `test_legacy_disclosure.py` 13 kontrollü Streamlit AppTest içerir.
  BL-003/ADR-0005: `test_market_data.py` 13 provider, `test_data_fetcher.py` 5 gerçek
  adaptör, `test_rss_failure_paths.py` 10 kaynak-kesinti ve `test_rss_transport.py`
  22 transport/tanı testi içerir. Son onaylı dilimde 31 yeni test eklendi.
- **Integration:** Yukarıdaki çevrimdışı API testleri provider yerine fixture
  kullanır. `make smoke` yerel stack/HTTP entegrasyonudur; canlı sağlayıcı
  doğrulamasıyla çevrimdışı birim test aynı kalite kanıtı sayılmaz.
- **End-to-end:** `scripts/run_dev_smoke.py` içinde temel UI kontratlarının yanında,
  kritik veri akışları için küçük kapsamlı uçtan-ucuna API senaryoları da eklenmiştir
  (watchlist ve paper ekleme/silme döngüleri). İsteğe bağlı olarak, `make smoke-ui`
  ile tarayıcı üzerinden temel UI e2e smoke senaryosu çalıştırılır.
- **Deterministik UI:** Çalışan yerel frontend ile
  `node scripts/run_frontend_ui_smoke.mjs --deterministic --front-base http://127.0.0.1:3010`
  veya `make smoke-ui-deterministic`. Backend gerekmez. Başlangıç komutları README
  10.3'te; Windows'ta kurulu Edge için `--browser-channel msedge` kullanılabilir.
  Bu hedef mevcut CI'ye otomatik eklenmedi; `make ci` başarısı bu on senaryoyu
  çalıştırdığı anlamına gelmez.
- **Security:** Secret taraması ve dependency audit henüz yapılandırılmadı.
- **Syntax/build minimum:** Repository içindeki her `.py` dosyasını `compile()`
  ile kontrol et; Streamlit için ayrı build adımı N/A.
  Next.js için `make build-frontend` vardır; yalnızca backend test/belge
  değişikliğinde frontend build yeniden koşulmadıysa açıkça belirtilir.
- **CDSK:** `python3 /Users/fatih/Dev/cdsk/scripts/validate_project.py .`
  Bu upstream validator yolu mevcut Windows ortamında yoktur; bu oturumda
  `.cdsk.json` içindeki 39 dosyanın dolu olması, sürüm ve 11 Project Boot
  bölümü yapısal olarak kontrol edildi. Upstream validator geçti denmez.
- **Transition:** Backend API contract, cron schedule ve health endpoint için ek gate.

### Windows doğrulama ortamı (2026-11-05)

- Sistem `python`/`py` alias'ları erişilemedi; uygulamayla gelen Python 3.12.14
  ile `.venv` oluşturuldu (bundled paketleri kullanır, Git tarafından ignore edilir).
  Eksik FastAPI, yfinance, ta ve httpx yalnız test ortamına kuruldu; requirements
  veya sürüm politikası değişmedi. Bu ortam uygulama runtime'ına bağlıdır.
- Test komutu: `.\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -p 'test_*.py'`.
  API TestClient, Windows sandbox'ında yerel socketpair kurarken takıldı;
  sandbox dışında hem normal hem dış socket bağlantıları engellenmiş koşuda
  60 test geçti. asyncio'nun loopback self-pipe bağlantıları serbest bırakıldı.
- Test gücü: Doji/FVG/RSI eşik operatörü ve sinyal çoğunluk kuralı bellekte
  bozuldu; dört varyant da assertion hatasıyla yakalandı. Üretim dosyaları değişmedi.
- 39 Python dosyası `compile()` ile kontrol edildi; bytecode yazılmadı.
  Starlette/httpx deprecation uyarısı var. Önceki Python 3.9/LibreSSL kaydı
  tarihsel kalır; bu ortamda aynı uyarı görülmedi. Patern/sinyal adımında frontend
  build ve canlı UI smoke çalıştırılmadı; sonraki UI doğrulaması aşağıdadır.

### İlk deterministik UI doğrulaması (2026-11-05, BL-002 öncesi)

- Frontend `package-lock.json` ile kuruldu; Next.js/React/Playwright sürümleri
  ve üretim kaynakları değişmedi. Next.js 14.2.18 kurulum güvenlik uyarısı sürüyor;
  bu çalışma dependency audit veya güvenlik güncellemesi değildir.
- Playwright 1.62.1 ve yerel headless Edge ile 5/5 senaryo hem `next dev` hem
  başarılı `next build` ardından `next start` üzerinde geçti; API/dış sağlayıcı
  sunucusu çalıştırılmadı. Browser context'lerinde beklenmeyen dış istek yok.
- Node sözdizimi kontrolleri iki script için geçti. Kalıcılık davranışı yalnız
  test tarayıcısında bozularak yeni senaryonun hatayı yakaladığı doğrulandı;
  Playwright arama yollarında bulunmadığında deterministik mod exit 1 verdi.
- Canlı UI smoke yeniden koşulmadı; yalnız yenileme sonrası response subscription
  sırası düzeltildi. Backend üretim/test kaynakları bu adımda değişmedi; 60 test
  için önceki doğrulama kaydı korunur, bu frontend adımında yeniden koşulmadı.
- Sınırlar: analiz fixture'ı mevcut UI'nın tükettiği alanları sınar; tam analiz
  API sözleşmesinin yerine geçmez. Boş RSI sonucunda mevcut metin hâlâ
  “Henüz tarama yapılmadı.”; iç içe FastAPI hata ayrıntısı yerine “İstek hatası”
  gösterilir. Bu mevcut ürün metni/hata sunumu borcu ayrıca değerlendirilir.

### BL-002 uygulama doğrulaması (2026-11-05)

- A metinleri/ADR-0004 kullanıcıca onaylandı. 14 RSS/NLP testi; eksik/geçersiz/
  naif/offset'li tarih, gerçek anla sıralama, sıfır/hata/nonfinite skor,
  karma/skorsuz/boş özet ve sembol filtre sözleşmelerini ağsız doğrular.
- 8 AppTest, gerçek `show_news_feed` / `show_ai_predictions` fonksiyonlarını AST
  ile alıp Streamlit içinde kontrollü RSS/ML/IO ile render eder. Monolit startup
  yan etkileri çalışmaz; bu tam uygulama browser testi değildir. Nullable
  skor/filtre/özet, yayın/çekilme zamanı, üst ML uyarısı, geçerli sayısal çıktı
  ve geçersiz yayılımda senaryo/bant/uyum/risk gizleme sınanır.
- 82 test birleşik olarak dış socket bağlantıları engellenmişken geçti;
  Windows asyncio için loopback serbest. API socketpair/AppTest temp dosyaları
  nedeniyle sandbox dışında koşuldu. Requirements politikası değişmedi;
  mevcut Streamlit/Plotly/NLP kütüphaneleri yalnız yerel `.venv` içine kuruldu.
- 10/10 cockpit senaryosu üretim ekranında geçti. İlk red mevcut ekranda eksik
  uyarıyı; ek red bozuk metadata'nın sahte sıfır yaş/1970 tarihi olmasını yakaladı.
  Son Next.js build, iki Node sözdizimi ve 41 Python dosyası compile kontrolü geçti.
- Bağımsız incelemede Important/Critical bulgu yok. Önerilen metadata tipi
  guard'ı red→green ile eklendi. Model ufuk/tarih ve etiket/filtre eşik borçları
  değiştirilmedi. Canlı RSS/Yahoo/SMTP ve tam legacy browser koşulmadı.
- Next.js güvenlik riski; Starlette/httpx, Streamlit `use_container_width` ve
  bare-context uyarıları açık. CDSK upstream validator ortamda yok; yapı/bağlantı
  kontrolü onun yerine geçmiş sayılmaz.

## Test Data and Isolation

### Accepted ADR-0006 OTT doğrulaması (2026-11-06)

- 10 contract ve 6 entegrasyon testi eklendi; önceki 2 runtime testi korunur.
  Onaylı literal fiyat/yön, iki yön/eşitlik/clamp, default/custom config,
  boş/tek gözlem, gapped timezone index ve caller/önceki çıktı korunması sınanır.
  Gerçek üretici→crossover/pozisyon okuyucuları mock olmadan test edilir.
- İlk RED: 9 contract testi 26 assertion failure/5 error, 6 entegrasyon testi
  13 missing-field assertion failure/1 empty IndexError. Sonraki kompleks fiyat
  3 error ve temporal fiyat 2 error reproducer'ları ValueError guard'larıyla
  green oldu. İlk yön NaN; NaN→±1 geçişi crossover sayılmaz.
- 18 geçerli transient mutant seed/VAR/config/max-min/eşitlik/sinyal/yön ve
  geçersiz fiyat hatalarını assertion failure ile yakaladı; error sıfır.
  Runtime testleri ayrı dar uyumluluk kanıtı olarak adlandırıldı.
- Bağımsız reviewer ilk 17, temporal edge düzeltmesi sonrası 18 hedef testi
  çalıştırdı; Important/Critical/eyleme dönük minor bulgu yok. Onaylı sözleşme
  veri alanıyla tamamlandı; yeni endpoint, dependency sürümü veya puan ağırlığı yok.
- Son birleşik 248 test dış socket engelli geçti; Windows asyncio loopback
  serbesttir. Önceki 232 testin tamamı bu yeni koşuda yer aldı.
- Boş/tek gözlem crossover okuyucusunda sınanır; mevcut pozisyon okuyucusunun
  asgari veri sınırı değiştirilmedi. Frontend/canlı servis tekrar koşulmadı.
  Ayrıntı `docs/20-execution/2026-11-06-adr0006-implementation.md` içindedir.

### RP-01–05 dar düzeltme doğrulaması (2026-11-06)

- Alert regresyonları önce 14 assertion failure (8 test), sonra eski 40 testle
  birlikte 48/48 green. RSI/MACD sıfır/None/NaN ve gerçek düşen RSI zinciri,
  güçlü SAT mevcut satış boyutu ve güçlü AL korunması sınandı; ağırlıklar aynı.
- DayTrader 10 test önce 43 assertion/subtest failure, sonra eski 17 ile 27/27
  green. Eksik veri, sonlu/pozitif olmayan entry fiyatı/ATR ve desteklenmeyen
  yön ValueError; indicator işi, saat ve kayıt mutasyonu öncesinde reddedilir.
  Gerçek scan bir hatalı sembol sonrası gerçek analyzer ile geçerli BUY döndürür.
- OTT 2 runtime testi 4 TypeError üretti, `.ffill()` sonrası eski 17 teknik
  testle 19/19 green. Bu yalnız pandas API uyumluluğu/index/girdi korunması
  kanıtıdır; seed veya yön matematiğinin doğru olduğu iddia edilmez.
- Ana ajan 94 hedef ve birleşik 232 testi dış socket engelli geçti (asyncio
  loopback serbest). Bağımsız reviewer 66 yeni+korunmuş risk/pozisyon/DayTrader
  testi çalıştırdı; Important/Critical/eyleme dönük minor bulgu yok.
- OTT seed/ott_trend sözleşmesi Accepted değildir; ADR-0006 tablosu bağımsız
  Fraction hesabıyla review edildi. Canlı sağlayıcı ve frontend tekrar koşulmadı;
  manifest/sürüm/puan ağırlığı değişmedi. Ayrıntı `docs/20-execution/2026-11-06-risk-position-fixes.md`.

### BL-001 risk/pozisyon doğrulaması (2026-11-06)

- 13 birleşik risk testi: RSI/MACD/EMA/hacim/destek-direncin katkıları, 2/4/7
  toplam puan sınırları ve altları, stop/hedef tutarları, sample std ile 252 gün
  yıllıklandırma, 20 getiri penceresi ve 100 dışı fiyat yuvarlaması. Gerçek
  hesaplayıcı zinciri ve bağımsız çağrı/girdi korunması ayrıca sınanır.
- 16 pozisyon testi: ayrı kural girdileri, iki yönlü toplam/karar/confidence,
  uyarılar ve geçerli boyut dalları. Gerçek RSI/MACD/EMA/MA200/BB/SuperTrend/VWAP
  zinciri 8.5 boğa - 2 ayı = 6.5 verir. OTT trend kuralı açık sentetik seriyle
  test edilir; gerçek OTT hesabı mevcut runtime'da hata verdiği için zincir
  kanıtına dahil edilmez. Güçlü SAT hatalı boyutu golden assertion değildir.
- 17 DayTrader testi: gerçek BUY/SELL/HOLD zincirleri, ATR gap/window/period,
  giriş offset'i ve tam duyarlıklı R:R, üç ondalık seviyeler, hacim ortalaması,
  yerel BUY/SELL miktar P&L ve aktiften geçmişe kayıt/tarih/durum aktarımı.
  İki testte açık seri taşıyıcısı yalnız consumer eşiklerini ayırır; gerçek
  indikatör zincirinin yerine geçtiği iddia edilmez. Saat sabit, I/O engellidir.
- 32 risk + 38 pozisyon + 34 DayTrader = 104 geçerli bellek içi mutant assertion
  failure verdi, error yok. İlk 19-getiri ve 19-hacim penceresi mutantları fixture
  açığı gösterdi; asimetrik şok/yakın eşik girdileri eklendikten sonra yakalandı.
- Bilinen yanlış davranışlar ayrı reproducer'larla raporlanır; passing golden
  testlere dönüştürülmez: RSI/MACD sıfır atlama, güçlü SAT sizing, OTT TypeError/
  alan uyumsuzluğu ve doğrudan kısa/boş/geçersiz girdi yolları. Ayrıntı BACKLOG
  RP-01–05 ve `docs/20-execution/2026-11-06-risk-position-tests.md` içindedir.
- Finansal formül/config/dependency değişmedi. Canlı sağlayıcı/frontend tekrar
  koşulmadı; genel BL-001/M1, runtime ve veri doğrulama açıkları korunur.

### BL-001 BOS/kombinasyon doğrulaması (2026-11-06)

- 15 BOS testi gerçek swing/kırılma zinciriyle iki yönü, kapanış eşitliği ve
  wick-only negatifleri, hacim %20 eşiğini, güç hesabı/cap/boost sırasını,
  swing güç/hacim/mesafe eşiklerini, lookback sınırını ve veri korunmasını sınar.
  TechnicalAnalyzer config etkisi gerçek yayımlanan sonuç üzerinden doğrulanır.
- 12 kombinasyon testi gerçek OHLCV ile iki yönde FVG+OB/FVG+BOS ve analyzer
  çıktılarını doğrular. Ayrı matcher testleri gerçek dedektörlerden üretilmiş
  kontrollü cache kayıtları kullanır; bunlar tek OHLCV zinciri kanıtı sayılmaz.
  Strict %2 OB yakınlığı, 3 günlük dahil zaman sınırı, fiyat/güç eşitlikleri,
  karşıt yön/uzak fiyat reddi ve confidence sırası literal beklentilerle sınanır.
- 25 BOS + 15 geçerli kombinasyon mutantı assertion failure üretti, error yok;
  gerçek kod geri geldiğinde 27 hedef test geçti. Dönem/yön/eşik/mapping/güç ve
  tarih/fiyat cezaları gibi gerçekçi hatalar yakalandı; disk üretim kodu değişmedi.
- Birleşik 166 test dış socket engelli geçti; Windows asyncio loopback serbest.
  Bağımsız incelemede önemli/engelleyici bulgu yok; bir mum sapmış belge ifadesi
  düzeltildi: sondan 10. mum dahil, daha yeni 9 mum hariç. Mevcut batch/future-
  neighbor ve sezgisel güç/confidence davranışı finansal strateji onayı değildir.
- Frontend/canlı sağlayıcılar yeniden koşulmadı; genel BL-001/M1, risk/pozisyon/
  ML ve ek yapı kenarları açık. Kanıt `docs/20-execution/2026-11-06-bos-combo-tests.md`.

### BL-001 değişken MACD/Bollinger doğrulaması (2026-11-06)

- 5 yeni test: MACD 12/26 dönemleriyle iki yönlü şok/reversion, 9 dönem sinyal
  ve histogram, 25/26 ve 33/34 gözlem görünürlük sınırları; Bollinger 20 fiyat
  penceresi, ±2 popülasyon std, aykırı değerin pencereden çıkması ve 19/20 sınırı.
  Seriler tarih index'ini korur; mevcut girdi korunması testi Bollinger'ı da kapsar.
- Beklenen MACD değerleri geometrik EMA sönümünden bağımsız Fraction hesabıyla
  türetildi ve literal tabloya yazıldı. Bollinger ortalama/varyansları elle
  doğrulanır. Testler beklenen değeri `ta`, pandas ewm/rolling veya gerçek
  analiz/config çıktısından üretmez; gerçek `add_indicator` çalışır.
- 12 bellek içi mutantın tümü assertion failure ile yakalandı: MACD fast/slow/
  signal dönemleri, çizgi/histogram işareti, sinyal alanı, erken sinyal üretimi;
  Bollinger dönem/std çarpanı, bant yönü, erken bant üretimi ve örneklem std.
  Üretim dosyası diskte değiştirilmedi; gerçek kodla 17 hedef test yeniden geçti.
- Birleşik 139 test dış socket engelli geçti (asyncio loopback serbest). Canlı
  sağlayıcı, frontend build/UI yeniden koşulmadı; finansal davranış/dependency
  politikası değişmedi. BL-001/M1'in kalan BOS/kombolar/risk/ML kapsamı açık.
- Plan/kanıt: `docs/20-execution/2026-11-06-macd-bollinger-tests.md`.

### ADR-0005 onaylı uygulama doğrulaması (2026-11-05)

- 134 birleşik backend testi dış socket bağlantıları engelliyken geçti; Windows
  asyncio için yalnız loopback serbest. Gerçek HTTP/sleep, Yahoo/RSS/SMTP yok.
- RSS 22 yeni test + korunmuş 24 ADR-0004/kesinti testiyle 46/46 geçti.
  Connect/read 8/8, üç toplam deneme, 1.5/3.0 sleep, yalnız onaylı retry nedenleri,
  429/503 Retry-After erteleme, final redirect sonucu ve response kapanışı sınandı.
  Parser istisnası/bozo/boş sonuç transport tekrarını tetiklemez. Önceki IO
  fixture'ları requests byte sınırına taşındı; haber/NLP/tarih assertion'ları korundu.
- Yahoo'ya 4 tanı testi eklendi: başarı, toparlanma, gözlenen hata sınıfları,
  tüm çağrının monotonic süresi ve yeni fetch reset/eski snapshot izolasyonu.
  Provider/adaptör toplam 18/18; mevcut retry/dönüş/exception sözleşmesi korundu.
- Legacy'ye 5 AppTest eklendi (toplam 13): failed/partial uyarısı, success/empty
  yanlış uyarı yokluğu, boş/simge filtreli liste ve bağımsız özet/liste tanıları.
  Gerçek fonksiyonlar kontrollü IO ile render edilir; tam monolit browser değildir.
- Yeni RSS sözleşmesi önce 41 assertion + 5 eksik alan hatası; close istisnası
  ek testi 1 failure; Yahoo 10 eksik alan hatası; legacy 7 assertion failure üretti.
  İlgili uygulamadan sonra tümü green. Frontend build/UI yeniden koşulmadı;
  önceki 10/10 üretim cockpit kaydı korunur. Canlı sağlayıcı smoke ayrı kalır.
- Ayrıntı ve inceleme: `docs/20-execution/2026-11-05-adr0005-implementation.md`.

### Önceki BL-003 hata-yolu doğrulaması (2026-11-05, ADR-0005 öncesi)

- Provider timeout iletimi ve gerçek BISTDataFetcher → `Ticker.history` zinciri
  önce 1 failure/3 error üretti; keyword iletimi düzeltildikten sonra geçti.
  Legacy çağrı override yokluğu ve OHLCV içeriğinin korunması ayrıca sınandı.
- Varsayılan 3 toplam deneme, 1.5/3.0 backoff, başarıda/final hatada ek sleep
  olmaması; timeout ve None/boş veri sonrası toparlanma/tükenme kontrollü testtir.
  Gerçek sleep ve ağ kullanılmaz. History timeout toplam işlem deadline'ı değildir;
  kurulu yfinance 1.8.0 lazy timezone ediniminde kendi timeout'unu kullanır.
- RSS timeout/disconnect, boş 503, kullanılabilir/boş bozo parse, tüm kaynakların
  kesintisi ve skorsuz özetler test edilir. Görünür HTTP hata gövdesi haber üreten
  beş subtestte red oldu; >=400 guard sonrası green. 301 metadata uyumu korunur.
  Feedparser nihai redirect durumunu örtebilir; yalnız bilinen durum guard'ı vardır.
- 24 hedef test ve birleşik 103 test dış socket engelli geçti; asyncio loopback
  serbesttir. 43 Python dosyasının compile ve diff kontrolü geçti.
- Bağımsız incelemede Important/Critical bulgu yok; ADR-0005 enum tutarsızlığı
  belgede düzeltildi. Yeni RSS timeout/retry/tanı politikası hâlâ Proposed.
  Frontend build/UI ve canlı Yahoo/RSS/SMTP bu backend adımında koşulmadı.
- CDSK upstream validator ortamda yoktur; yapı/bağlantı kontrolü onun sonucu
  sayılmaz. Önceki dependency/deprecation ve M1 kapanış riskleri açık kalır.

Unit testler sabit seed ve küçük sentetik OHLCV fixture'ları kullanmalı; gerçek
ağ, saat ve global Streamlit state birim testlerden ayrılmalıdır. Gerçek finansal
kullanıcı verisi fixture yapılmamalıdır.

### Önceki BL-001 ML yardımcıları doğrulaması (2026-11-06, ADR-0007 öncesi)

Bu ilk test diliminin tarihsel kaydıdır; sonraki ADR-0007 düzeltmeleri aşağıdadır.

- `test_ml_helpers.py` (9): safe_divide fallback/broadcast, batch cleaner
  clip/ffill/quantile, numeric tanı, yalnız geçerli target prefix'i, confidence
  helper'ın kendi nested girdi sınırı ve eğitimsiz hata. Native target kuyruğu
  veya fiyat sözlüğüyle helper uyumu passing beklentiye dönüştürülmedi.
- `test_simple_ml_predictor.py` (15): gerçek feature oran/lag/pencere/optional
  kolonlar, 30 geçerli hedef eşiği, zaman sıralı split ve gerçek RF50 eğitimi.
  Sabit %2 hedefle gerçek fit→predict ilişkisi sınanır; tam forest çıktısı pinlenmez.
  Ayrı estimator boundary double'ı yalnız kesin getiri/fiyat/eşik/cache aritmetiğini
  sınar. Importance ve NaN/stale state kusurları ayrı reproducer'dır.
- `test_ml_features_training.py` (16): 30 baz kolon, rolling/sample std/lag ve
  optional seri sınırları; gerçek StandardScaler/MinMaxScaler/LinearRegression
  ve DummyRegressor baseline ile nonzero RMSE/MAE/R². Target sınırı bağımsız
  etiketlerle ve terminal NaN ile değiştirilir; feature/clean/scaler/fit/metric
  gerçektir. Hata izolasyonunda fit-failure doubles ayrıca açıklanır. Native
  hedefle dört varsayılan model ve predict_price entegrasyonu doğrulanmış değildir.
- 22 helper + 23 simple + 35 advanced = 80 geçerli bellek içi mutant assertion
  failure verdi, error yok; kod diskte değiştirilmedi. Restore sonrası 40 test
  green, bağımsız reviewer 40 testi yeniden koştu; eyleme dönük bulgu yok.
  Birleşik 288 test dış socket engelli geçti (Windows asyncio loopback serbest).
- Test ortamı Python 3.12.14, numpy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1,
  SciPy 1.18.1, joblib 1.6.0. Eksik sklearn mevcut gereksinimden yalnız ignore
  edilen .venv'e kuruldu; manifest/pin/support policy değişmedi. İlk native
  SciPy importları uzadı ve sonunda tamamlandı; alternatif .venv/ml-compat
  SciPy 1.16.3 klasörü hiçbir teste/sys.path'e alınmadı.
- 57 Python compile ve CDSK 39 belge/11 Boot bölümü yapısal kontrolü; bağlantı,
  ADR ve diff kontrolleri yapıldı. Upstream validator ortamda yok; bunlar onun
  sonucu değildir. Frontend/canlı sağlayıcı yeniden koşulmadı. BACKLOG ML-01–08
  açıktır; test sayısı finansal doğruluk/kalibrasyon veya M1 kapanışı kanıtı değildir.
  Plan/kanıt: `docs/20-execution/2026-11-06-ml-helper-tests.md`.

### ADR-0007 kabul kapısı — Accepted ve uygulandı

- NaN hedef kuyruğu ile gerçek0 getiri ayrımı; min50 geçerli etiket ve orijinal
  kaynak satırlarında i+h<ilk test konumu; purge sonrası örnek sayısı/hata yolları.
- Sabit feature temizliğinde prefix korunması; train-only q01/q99 ve scaler,
  heldout/inference verisinde yeniden fit olmaması. Sabit split ile değişen heldout
  değerler aynı bounds/model girdisini fit sırasında etkilememelidir.
- Her dört gerçek estimator için aynı scaled X ve inverse target; literal fiyat
  oracle'lı ayrı label boundary örneği ve native hedefle tam smoke birbirinden ayrılır.
- Failed/partial retrain paketi tutarlı; güvenilen temp artifact roundtrip,
  sürümsüz/bozuk paket rejection ve başarısız load'da mevcut state korunması.
- Tasarım aşamasında40 test tekrar geçmişti. Onaylı uygulamada
  `test_ml_contract.py`19 + `test_ml_artifacts.py`11 =30 yeni test; toplam70 ML.
  Korunan9 helper/16 training/15 simple ile finite/NaN hedef, original-position
  purge, train-only bounds ve tüm modellerin gerçek dönüşümü doğrulanır.
- 19 contract testi önce29 assertion/subtest failure ve7 eksik sözleşme hatası;
  ilk9 artifact testi sandbox dışındaki geçerli RED koşusunda43 failure/7 hata.
  Temp sandbox erişim hatası ayrı çevre engelidir, davranış RED'i sayılmadı.
  Sonraki üç review regresyonu7 failure+1 OverflowError → green; sklearn'in
  integer-valued float sample sayısı uyum hatası10 error'dan sonra düzeltildi.
- 15 geçerli transient mutant assertion failure verdi; ilk reset metni yanlış
  eşleşen deneme sayılmadı, düzeltilen mutant yakalandı. Disk mutation yok.
  Bağımsız ilk68 test incelemesinde bulunan scaler coherence/huge-horizon kusurları
  kapandı; re-review70 ML testi geçti (9.037 s), kalan eyleme dönük bulgu yok.
- Son318 backend testi dış socket engelli geçti (19.869 s, Windows asyncio
  loopback serbest). Trusted temp joblib gerçek I/O, native dört-model smoke ve
  ayrı .025 label-boundary160→164 oracle'ı kullanılır; accuracy/kalibrasyon yok.
  59 Python compile, CDSK39 belge/11 Boot bölümü ve link/ADR/diff kontrolü;
  upstream validator ortamda yok, yapısal kontrol onun sonucu değildir.
  Frontend/canlı sağlayıcı yeniden koşulmadı. Ayrıntı
  [execution kaydı](../20-execution/2026-11-06-adr0007-implementation.md).

### Simple ML-04–06 düzeltmeleri (2026-11-06)

- `test_simple_ml_fixes.py`13 yeni regresyon: gerçek RF başarılı→short/prep/fit/
  metric hatası veya nonfinite metric; refit ve importance10 baz/partial optional
  eşlemesi. Explicit estimator boundary yalnız invalid getiri/current price/
  overflow ve cache exception/retry durumlarını ayırır, forest hesabı sayılmaz.
- İlk13 test29 assertion/subtest failure verdi, error yok. Düzeltmeyle13 yeni +
  15 korunmuş simple +9 helper=37 hedef test geçti; bağımsız reviewer37 testi
  tekrar çalıştırdı (0.877 s), eyleme dönük bulgu yok.
- Reset flag/cache, metric validity, fitted-name mapping/weight order, duplicate
  metadata ve türetilmiş percent/pozitif fiyat için11 geçerli transient mutant
  assertion failure verdi; error yok. Kod diskte mutate edilmedi, restore13 green.
- Toplam83 ML/331 backend testi dış socket engelli doğrulandı (asyncio loopback
  serbest);60 Python compile,39 zorunlu belge/11 Boot bölümü ve yerel bağlantı/
  ADR/diff kontrolü. Upstream validator ortamda yok; yapısal kontrol onun yerine
  geçmez. Frontend/canlı sağlayıcı tekrar yok, genel accuracy/M1 kapısı açık.
- Yalnız Simple üretim modülü değişti; advanced/config/dependency dosyaları
  başlangıç SHA256 ile aynı. RF50/depth10/seed42, feature formülleri, split/min30,
  strict±.01 sinyal ve confidence/percent hesapları korunur. Yeni ADR/persistence
  schema yok. Kanıt [execution kaydı](../20-execution/2026-11-06-simple-ml-fixes.md).

### Git teslimi ve ADR-0008 hazırlığı (2026-11-06)

- `abfc592` için331 backend testi dış socket engelli yeniden geçti (24.007 s,
  Windows asyncio loopback serbest). Next.js üretim build/lint/type ve üretilmiş
  frontend üzerinde Edge10/10 deterministik senaryo başarılı; dış istek0.
  Yalnız test için açılan yerel server kapatıldı; canlı sağlayıcı kullanılmadı.
- ML-07 gerçek LR/gerçek scaler + bağımsız .40 target boundary ile h1/2/3/7/8/
  14/30/31 cap tablosu; ML-08 float fiyat sözlüğü TypeError ve stale-date source
  aritmetiği yeniden üretildi. Üretim kodu veya geçerli testler değiştirilmedi.
- ADR-0008 Proposed kabul ölçütleri: tek cap uç/yön/eşitlik testleri, fiyat/
  legacy/invalid helper payload/context/strict threshold ve mutation korunması;
  kontrollü AppTest gözlem selector/kart/caption ve0..h şematik grafik. Bunlar
  yeni test başarısı iddiası değildir; özel onay sonrası implementation/TDD.
- Bağımsız builder review planlama için önemli açık soru bulmadı. Tasarım/teslim
  [kaydı](../20-execution/2026-11-06-ml-horizon-confidence-design.md); upstream
  CDSK validator yokluğu ve genel M1/accuracy/dependency kapıları sürer.

### Accepted ADR-0008 doğrulaması (2026-11-06)

- `test_ml_horizon_confidence.py`14 yeni test: literal interval uçları/huge h,
  iki yön/eşitlik/skip/ensemble, shared-helper consumer, price/legacy/invalid
  payload/context/strict thresholds ve ratio overflow. Estimator boundary +
  gerçek identity scaler ayrı tanımlanır; native dört-model smoke korunur.
- `test_legacy_ml_horizon.py`4 yeni controlled AppTest: raw numeric selector,
  stale data last_index/kesin tarih yokluğu, bütün UI h caption'ları ve0 anchor/
  0..h line/1..h bands. Gerçek extracted panel, fake IO/estimator ve gerçek
  bounds helper kullanır; full monolith/browser/live-provider kanıtı değildir.
- İlk14 core test9 assertion failure/61 eksik API-sözleşme hatası; ilk4 panel
  testi7 assertion failure/0 error verdi. Son42 core/17 AppTest hedefleri green;
  bağımsız reviewer59 testi yeniden koştu (7.610 s), önemli bulgu yok.
- Fixture artık gerçek response gibi ensemble içerir:108/112/110 için mean110,
  std=sqrt(8/3); production UI spread formülü değişmedi. Eski numeric oracles
  buna göre düzeltildi; testler helper'ın UI skoruna bağlanmasını da reddeder.
- 16 core+3 UI=19 geçerli bellek içi mutant assertion failure verdi; error yok.
  İlk numeric error-key mutantı kaçtı; payload{"error":103.} eklendi ve yakalandı.
  Disk mutation yok, restore hedefler green. Genel mutation coverage iddiası yok.
- Birleşik349 backend testi dış socket engelli;97 ML core/17 controlled Streamlit.
  62 Python compile,39 zorunlu belge/11 Boot bölümü, Accepted ADR1–8/link/diff.
  Upstream validator ortamda yok; yapı kontrolü onun sonucu değil. Next.js kaynakları
  aynı olduğundan build/browser tekrarı yok, canlı sağlayıcı çalıştırılmadı. Ayrıntı
  [execution kaydı](../20-execution/2026-11-06-adr0008-implementation.md).

## Release Gates

- P0 test başarısızsa release olmaz.
- Ağ yokluğu birim testleri engellemez; canlı smoke test “skipped/degraded” olarak
  ayrı raporlanır.
- Mock/gerçek veri karışması ve sessiz entegrasyon hatası release blocker'dır.
- CDSK validator, sözdizimi ve dokümantasyon/changelog kontrolü zorunludur.
