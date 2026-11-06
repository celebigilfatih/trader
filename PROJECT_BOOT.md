# BIST Teknik Analiz Uygulaması — Project Boot

- **Document version:** 0.1
- **CDSK version:** 0.1.0
- **Last updated:** 2026-11-06

> Bu belge güncel çalışma özetidir. Ayrıntılı ürün, mimari ve karar belgelerinin
> yerine geçmez; onlara bağlantı verir. Bilinmeyen alan `TBD`, geçersiz alan
> kısa gerekçeyle `N/A` bırakılır.

## Project Compass

- **North star:** BIST araştırma verisini, teknik analizi ve deneysel modelleri
  kaynağı ve riski açık, güvenilir bir karar-destek deneyiminde birleştirmek.
- **Current mission:** Mevcut prototipi ölçülebilir, test edilebilir ve güvenli
  bir ürün tabanına dönüştürmek.
- **Success signal:** Ana kullanıcı akışları deterministik testlerle doğrulanır;
  veri kaynağı/tazeliği ve mock/deneysel çıktı kullanıcıya açıkça görünür.
- **Accepted trade-offs:** MVP; kapalı beta + paper/simülasyon odaklı yürütülür; gerçek
  emir/işlem ve dağıtım SLA’ı bu aşamada hedeflenmez.

## 1. Project Identity

- **Project name:** BIST Teknik Analiz Uygulaması
- **One-line vision:** BIST hisselerini tek arayüzde açıklanabilir teknik analiz,
  tarama, haber ve deneysel tahminlerle araştırmak.
- **Problem statement:** Piyasa verisi, gösterge, tarama ve haber analizi farklı
  araçlara dağılmış; mevcut prototipin güvenilirlik ve ürün sınırları da kalıcı
  belgelerde tanımlanmamıştı.
- **Target users:** Kendi araştırmasını yapan bireysel BIST kullanıcısı/analisti; kapalı beta davetli erişim.
- **Success criteria:** Kapalı beta'da ana akışların belirli zamanlarda tutarlı ve
  deterministik çalışması; veri tazeliği + mock/provenance görünürlüğü; güvenli geçiş adımları.
- **MVP goal:** Günlük karar cockpit’i, paper portföy, alarm ve otomatik snapshot akışını çalışır kılmak; kalite temellerini inşa etmek.
- **Explicit non-goals:** Aracı kurum/emir entegrasyonu, otomatik emir, yatırım
  tavsiyesi/garanti, komisyon/nakit/gerçek muhasebe.

## 2. Current Status

- **Phase:** Kullanıcının devam talebiyle M1 geliştirmesi yeniden başladı
  (2026-11-05); ADR-0007 ve Simple ML-04–06 uygulamaları tamamlandı (2026-11-06).
- **Version:** Release etiketi yok (`TBD`).
- **Active sprint / milestone:** M0 CDSK başlangıcı tamamlandı; M1 güvenilirlik
  tabanı başladı. Sonraki adım API-first Next.js cockpit’i ile uçtan uca akışın
  stabilizasyonu.
- **Current focus:** ML-01–06 bilinen dar kusur dilimleri tamamlandı;
  83 ML/331 backend testi. Sonraki öncelik ML-07 ufuk cap/ML-08 confidence
  sözleşmesini somutlaştırmak ve kalan M1 kapanış eksikleri.
- **Critical risks:** Finansal çıktının yanlış yorumlanması; mock/gerçek veri
  karışması; dış servis kesintisi; ağ bağımlı ve yetersiz testler; `.next` gibi
  frontend build artefaktı yönetimi; açık veri lisansı/dağıtım modeli olmaması.
- **Blocking decisions:** Gerçek kullanıcı erişimi operasyon sahipliği,
  destek matrisi ve üretim kapıları. ADR-0004–0007 açık onayları uygulandı;
  ML cap/confidence/takvim politikası bu onaylara dahil değildir.
- **Last status update:** 2026-11-06

## 3. Repository Navigation

1. `AGENTS.md`
2. `docs/00-product/CONSTITUTION.md`
3. `PROJECT_BOOT.md`
4. `README.md`
5. `docs/00-product/PRODUCT_SPEC.md`
6. `docs/00-product/SCOPE.md`
7. `docs/00-product/ROADMAP.md`
8. İlgili `docs/10-architecture/` belgeleri
9. İlgili `docs/50-decisions/` ADR'leri
10. İlgili quality/operations belgeleri
11. `CHANGELOG.md`

## 4. Architecture Snapshot

- **Layers:** Streamlit presentation/orchestration → veri erişimi → teknik
  analiz/sinyal → tarama → deneysel ML/NLP → çıktı.
- **Main integrations:** Yahoo Finance, beş RSS kaynağı ve isteğe bağlı SMTP.
- **AI components:** Yerel scikit-learn tahminleri ve VADER/TextBlob sentiment; bazı sentiment akışları mock.
- **AI lab policy:** Core kararı etkilemeyecek deneysel katman.
- **Data/storage:** Kalıcı hedef: PostgreSQL; mevcut durumda bellek/session/cache ve
  `portfolio_data.json` ile çalışıyor.
- **Deployment model:** Hedef üretim modeli artık `FastAPI + Next.js + PostgreSQL`
  + Hostinger/Coolify olarak alınmış durumda; geçiş stratejisi strangler pattern.
- **Current architecture in use:** `backend/app/*` API servisi + `frontend/app/*`
  MVP cockpit. Streamlit (`run.py`) hali hazırda legacy destek olarak korunuyor.
- **Detailed sources:** `docs/10-architecture/OVERVIEW.md` ve ilgili belgeler

## 5. Decision Snapshot

| Decision | Status | ADR / source |
|---|---|---|
| CDSK 0.1.0 repository standardı kullanılır | Adopted by explicit user request | `.cdsk.json`, `AGENTS.md` |
| Ürün hedefleri kapalı beta, paper portföy ve güvenli sınırlar ile tanımlandı | Accepted | `ADR-0001`, `docs/50-decisions/ADR-0001.md` |
| Mimari hedefi API-first FastAPI/Next.js olarak konumlandırıldı | Accepted | `ADR-0002`, `docs/50-decisions/ADR-0002.md` |
| Kimlik doğrulama ve bildirim modeli MVP için netleştirildi | Accepted | `ADR-0003`, `docs/50-decisions/ADR-0003.md` |
| RSS yayın zamanı ile elde etme zamanı ve hesaplanamayan duygu skoru ayrımı | Accepted ve uygulandı | `ADR-0004`, `docs/50-decisions/ADR-0004.md` |
| RSS istek bütçesi ve geçici kaynak tanıları | Accepted ve uygulandı | `ADR-0005`, `docs/50-decisions/ADR-0005.md` |
| OTT seed ve yön serisi sözleşmesi | Accepted ve uygulandı | `ADR-0006`, `docs/50-decisions/ADR-0006.md` |
| ML hedef/ön işleme ve model artifact sözleşmesi | Accepted ve uygulandı | [ADR-0007](docs/50-decisions/ADR-0007.md) |
| 3D Web Experience profili uygulanmaz | N/A for current scope | `docs/70-experience/3D_WEB_EXPERIENCE.md` |

> Yalnızca en önemli güncel kararları özetle. Kararın tam gerekçesi ve sonuçları
> ADR'da kalır.

## 6. AI Context

- **Purpose:** Repository bağlamını koruyarak mevcut prototipi güvenilir bir
  ürün temeline taşımak.
- **Constraints and prohibitions:** Yatırım tavsiyesi/garanti üretme; mock veriyi
  gerçek gösterme; secret/portföy verisini bağlama yazma; onaysız mimari,
  deployment veya kapsam kararı alma.
- **Open questions:** Python sürüm kısıtları, onaylı kullanıcı onboard operasyonu,
  veri saklama ve backup politikalarının operasyonal onayı.
- **Recent decisions:** ADR-0001–0006 kabul edildi. Kullanıcı somut ADR-0006
  onay sorusuna “onaylıyorum” dedi; başka algoritma/version/üretim onayı değildir.
  Sonraki somut ADR-0007 sorusuna “Onaylıyorum.” ile o sözleşme de Accepted oldu.
- **Recently completed:** Simple trained/cache lifecycle, fitted feature-importance
  adları ve finite/pozitif sonuç doğrulaması.13 yeni regresyonla83 ML/331 backend
  testi geçti;11 mutant yakalandı, bağımsız37 hedef test incelemesi bulgusuz.

## 7. Current Priorities

1. ML-07 cap, ML-08 confidence API ve UI takvim/gözlem farkı için somut
   politika/sözleşme tasarımı; finansal kural/API değişirse AGENTS onay kapısı.
   Simple ML-04–06 mevcut alan/formül sınırında tamamlandı; genel M1 ayrı.
2. Next.js cockpit’de auth, hata yönetimi, boş veri ve retry akışlarını sağlamlaştırmak.
3. Paper portföy ve karar cockpit’i için write operasyonlarını (ekleme/çıkarma) ve
   senaryo testlerini eklemek.
4. Diğer Yahoo yolları ve üretim gözlem ihtiyacını kapsamlandırmak; mevcut
   OHLCV/RSS tanıları yereldir, toplam SLA veya merkezi alarm değildir.
5. Davetli beta kimlik ve in-app/e-posta alarm hattını açmak.

## 8. Known Constraints

- 2026-08-19 itibarıyla, `make ci` akışı backend çalışabilirlik ve smoke
  kontrolünü geçebilecek şekilde stabilize oldu; çevrimdışı deterministik test ve
  veri güvenilirliği açmazları halen izleniyor.
- Frontend lockfile ile yeniden kurulduğunda Next.js 14.2.18 güvenlik uyarısı
  tekrar görüldü; sürüm ve dependency politikası değişmedi, audit yapılmadı.
- pandas 3.0.1 OTT runtime, seed ve yön entegrasyonu Accepted ADR-0006 ile
  düzeltildi. Bu mevcut prototip recurrence'idir; canonical OTT veya finansal
  etkinlik kanıtı değildir. Genel bağımlılık destek matrisi/audit tamamlanmadı.
- ADR-0007 ML-01–03/advanced ML-04 ve sonraki Simple ML-04–06 dilimi kapandı.
  Ufuk cap/confidence/takvim uyumsuzlukları BACKLOG'da açıktır.
  Sürümsüz joblib yeniden eğitim ister; format kontrolü deserialization güvenliği
  veya dependency sürümleri arası uyumluluk değildir. Model accuracy kanıtı yoktur.
- Repository geçmişten kalan `__pycache__/*.pyc` dosyalarını izliyor.
- Constitution `Draft`; bağlayıcı hale gelmesi açık ürün sahibi onayı gerektirir.
- Windows'ta `python`/`py` alias'ları çalışmadı. Testler uygulamayla gelen
  Python 3.12.14 ve ignore edilen `.venv` üzerinden çalıştırıldı. Sandbox'ta
  asyncio'nun yerel `socketpair` kurulumu takıldığı için API testleri sandbox
  dışında doğrulandı; dış ağ engeli korunarak yalnız loopback'e izin verildi.
- Belgelenen CDSK validator'ın macOS yolu bu ortamda yok. 39 zorunlu belge ve
  11 Project Boot bölümü yapısal olarak kontrol edildi; bu upstream validator
  sonucu değildir.

## 9. Working Agreements

- Önce repository ve dokümantasyon doğrulaması.
- Küçük, doğrulanabilir adımlar.
- Kritik karar için ADR ve açık onay.
- Her anlamlı değişiklikte `CHANGELOG.md`.
- Ayrıntılı kaynağı Project Boot içine kopyalamak yerine bağlantı verme.
- Finansal veri kaynağı/tazeliği ve mock/deneysel niteliğini açık tutma.

## 10. Exit Checklist

### Definition of Ready

- [x] Amaç ve kabul kriterleri açık.
- [x] İlgili belgeler ve mevcut kod incelendi.
- [x] Belirsizlikler ve onay kapıları işaretlendi.
- [x] ADR ihtiyacı değerlendirildi.
- [x] Doğrulama yöntemi tanımlandı.

### Definition of Done

- [x] CDSK uygulama kabul kriterleri karşılandı.
- [x] Önceki CDSK temel tesliminde validator ve `make ci` başarısı kayıtlıdır;
  güncel Windows ortamındaki doğrulamalar ve sınırlar aşağıdaki handoff'tadır.
- [x] İlgili dokümantasyon güncellendi.
- [x] ADR ihtiyacı ve changelog tamamlandı.
- [x] Repository tutarlılığı doğrulandı.
- [x] Açık riskler ve sonraki adım kaydedildi.
- [x] Çevrimdışı market-data/API/teknik hesaplama, patern/sinyal ve RSS/NLP ile
  kontrollü Streamlit, ADR-0005 transport/tanı ve değişken MACD/Bollinger testleri
  ile temel BOS/kombolar ve risk/pozisyon/DayTrader testleri eklendi; toplam
  backend test sayısı331;83 ML testini ve13 yeni Simple regresyonunu içerir.
  Bu, M1'in bütün kabul kriterlerinin kapandığı anlamına gelmez.
- [x] On deterministik tarayıcı senaryosu üretim frontend'inde ve Next.js
  üretim derlemesi geçti; canlı sağlayıcı smoke'u bu adımda çalıştırılmadı.

## 11. AI Handoff

### Güncel devir — 2026-11-06

- **Yapılan:** Simple ML-04–06 mevcut çıktı biçimleriyle düzeltildi. Her train
  eski flag/cache'i siler; nonfinite metric success değildir. Importance gerçek
  RF fitted isimleri ve ağırlıklarını eşler. Inference invalid return/current/
  derived değer, overflow veya pozitif olmayan fiyat için error verir; eski son
  tahmin kalmaz, geçerli retry ve signal/confidence formülü korunur.
- **Kanıt:** 13 yeni test önce29 assertion/subtest failure (error0), sonra13+15
  simple+9 helper=37 hedef green.11 geçerli transient mutant assertion failure
  verdi, error yok; diskte mutation yok. Bağımsız reviewer37 testi yeniden geçti
  (0.877 s), eyleme dönük bulgu yok. Son331 backend testi dış socket engelli geçti
  (11.437 s, asyncio loopback serbest); toplam83 ML.60 Python compile,39 zorunlu
  belge/11 Boot bölümü, yerel link/ADR/whitespace/diff ve başlangıç hash kontrolleri
  geçti. Upstream validator ortamda yok; yapısal kontrol onun sonucu değildir.
- **Karar/sınır:** Yeni ADR/API/çıktı alanı/persistence/model/config/dependency yok.
  Advanced ADR-0007 dosyası başlangıç SHA256 ile aynı. Simple fiyat/getiri/percent
  hesapları, strict±.01 sinyal/confidence, RF50/depth10/seed42 ve split/min30 aynı.
  Aktif Simple UI tüketimi kaynakta bulunmadı; gerçek panel advanced kullanır.
  Frontend/canlı sağlayıcı tekrar yok, finansal accuracy/kalibrasyon iddiası yok.
- **Git/sonraki adım:** HEAD `6ed1a59`, önceki dirty değişiklikler korundu;
  commit/push yok. Sonraki iş ML-07/08/takvim kararını somutlaştırmak; finansal
  politika/API değişikliği gerekirse Proposed ADR ve açık onay. Genel BL-001/M1/M2,
  Constitution Draft ve dependency/operasyon kapıları açık. Plan/kanıt
  [Simple ML kaydındadır](docs/20-execution/2026-11-06-simple-ml-fixes.md).

### Önceki ADR-0007 uygulama kaydı — 2026-11-06

- **Onay/uygulama:** Kullanıcı somut ADR-0007 sorusuna “Onaylıyorum.” dedi;
  Accepted kaydedildi ve uygulandı. NaN hedef/source-position purge, stateless
  feature ve train-only bounds, ortak4-model scaler/inverse, coherent advanced
  state, format1 joblib. Mevcut model/feature formülü/cap/dependency aynı.
- **Kanıt:** 19 contract/11 artifact yeni test, toplam70 ML. İlk contract RED29
  failure/7 hata; artifact geçerli RED43/7.15 geçerli transient mutant assertion
  failure verdi (disk mutation yok). Review scaler coherence ve dev ufuk bulguları
  üç regresyonda7 failure/1 OverflowError→green kapandı. sklearn sample sayısının
  tam değerli float olma uyumu da düzeltildi. Son318 test dış socket engelli geçti
  (19.869 s, asyncio loopback serbest); reviewer70 testi yeniden geçti, açık bulgu yok.
- **Doğrulama/sınır:** 59 Python compile,39 zorunlu belge/11 Boot bölümü,
  link/ADR/whitespace/diff geçti. Upstream validator ortamda yok; yapısal kontrol
  onun sonucu değildir. Native4-model smoke ile literal160→164 label boundary
  ayrıdır; finansal doğruluk/kalibrasyon iddiası yok. Frontend/canlı servis tekrar yok.
  İlk artifact sandbox temp erişim hatası dış sandbox testle aşıldı; kullanıcı
  dosyası açılmadı/migrate edilmedi. Sürümsüz model yeniden eğitim gerektirir.
- **Git/sonraki adım:** HEAD `6ed1a59`, önceki dirty değişiklikler korundu;
  commit/push yok. Simple ML-04–06 dar düzeltmeleri; sonra ML-07/08/takvim
  sözleşme/politika ihtiyacı ve genel BL-001/M1/M2. Anayasa Draft ve önceki
  dependency/operasyon kapıları açık. Plan/kanıt
  [execution kaydındadır](docs/20-execution/2026-11-06-adr0007-implementation.md).

### Önceki ADR-0007 tasarım kaydı — 2026-11-06

- **Yapılan:** ML-01–03 için seçenek/etki/kabul ölçütleri ve geri dönüş tasarımı;
  ADR-0007 Proposed. Öneri NaN target, kaynak satırıyla ufuk purge, train-only
  %1/%99 bounds ve ortak model scaler/inverse, tutarlı advanced retrain state,
  format1 joblib. Eski sürümsüz paket yeniden eğitim gerektirir; uygulanmadı.
- **Kanıt/review:** Üç kusur sentetik veriyle tekrar üretildi: h2 terminal[0,0],
  .025 bilinen label/LR fiyat160 yerine164 beklenirken160; cleaner eski satır.2→.3.
  40 mevcut ML testi dış socket/joblib engelli geçti (1.013 s). Bağımsız builder
  incelemesi, belgede zaten yazılı ürün sahibi onayı dışında önemli soru bulmadı.
  57 Python memory compile, 39 belge/11 Boot bölümü, yerel link/ADR/whitespace
  ve üretim ML/test/manifest değişmezliği kontrolleri geçti. Upstream validator
  ortamda yok; yapısal kontrol onun sonucu değildir. Önceki288 full test kaydı
  korunur; bu belge adımında full backend/frontend/canlı servis tekrar koşulmadı.
- **Karar kapısı:** AGENTS mimari/veri modeli açık onayı gereği ADR-0007
  Proposed kalır. Genel “Sıradaki Adım” henüz görülmemiş bu artifact/ön işleme
  değişikliğinin onayı sayılmaz. Somut önerinin onayı uygulama için son kapıdır;
  yeni kod/dependency, kullanıcı artifact'ı/migration veya Accepted değişikliği yok.
- **Git/sonraki adım:** HEAD `6ed1a59`, önceki dirty değişiklikler korundu;
  commit/push yok. Onayda Accepted kaydı ve ayrıntılı görev planı; sonra hedef/
  purge, bounds, dört-model inference ve state/persistence regresyonları.
  ML-01–08 hâlâ açık, Simple/ufuk cap/confidence/takvim ve genel M1/M2 ayrı.
  Kaynak: [tasarım kaydı](docs/20-execution/2026-11-06-ml-correction-design.md)
  ve [ADR-0007](docs/50-decisions/ADR-0007.md).

### Önceki ML test kaydı — 2026-11-06

- **Yapılan:** ML helper/feature/eğitim için 3 yeni test dosyasında 40 test.
  Gerçek RF/LinearRegression/scaler/baseline ile bağımsız sayılar; estimator
  eşik/hata doubles'ı ve advanced target boundary açıkça ayrılır. Native hedef
  kuyruğu veya hatalı inference doğru beklenti olarak testlere yazılmadı.
- **Kanıt:** 22 helper + 23 simple + 35 advanced = 80/80 geçerli bellek içi
  mutant assertion failure, error yok; restore 40 green. Birleşik 288 test
  dış socket engelli geçti (asyncio loopback serbest). Bağımsız reviewer 40 testi
  yeniden çalıştırdı; eyleme dönük bulgu yok. 57 Python compile, 39 belge/11 Boot
  bölümü, bağlantı/ADR/whitespace ve ML üretim/manifest değişmezliği kontrolleri
  başarılı. Upstream CDSK validator yok; yapısal kontrol onun sonucu değildir.
- **Açık bulgular:** ML-01 native terminal target=0; ML-02 eğitim/inference
  scaler farkı; ML-03 future-dependent clipping. ML-04 stale retrain state,
  ML-05 simple importance adları, ML-06 NaN tahmin, ML-07 ufuk cap farkı,
  ML-08 confidence girdi uyumsuzluğu. Sentetik reproducer'lar execution kaydında;
  finansal etki/accuracy veya tüm dört model entegrasyonu doğrulanmadı.
- **Ortam/sınır:** Mevcut gereksinimden yalnız ignored .venv'e sklearn 1.9.1
  kuruldu; SciPy 1.18.1 kullanıldı. İlk native importlar sonunda tamamlandı;
  diagnostic .venv/ml-compat SciPy 1.16.3 test yoluna alınmadı. Manifest/pin,
  üretim ML kodu/policy/artifact, API veya yeni ADR değişmedi. Frontend/canlı
  sağlayıcı tekrar koşulmadı; dependency/deprecation ve Constitution Draft açık.
- **Git/sonraki adım:** Önceki değişiklikler korundu; HEAD `6ed1a59`, commit/push
  yok. Önce ML-01–03 için küçük düzeltme planı ve gerekli sözleşme/onay değerlendirmesi;
  sonra diğer ML bulguları/ek yapı testleri ve genel BL-001/M1/M2 kapanışı.
  Plan/kanıt `docs/20-execution/2026-11-06-ml-helper-tests.md` içindedir.

### Önceki ADR-0006 uygulama kaydı — 2026-11-06

- **Onay/uygulama:** Somut ADR-0006 sorusuna kullanıcı “onaylıyorum” dedi;
  Accepted kaydedildi. OTT ilk fiyatı loop öncesinde seed eder; eski ott/
  ott_signal alanları korunur, float ott_trend ilk gözlemde NaN/sonrasında ±1.
  Config/adjusted VAR/max-min recurrence/else eşitliği ve tüketici ağırlıkları aynı.
- **Girdi/okuyucular:** Boş veride üç boş Series, tek gözlemde fiyat/eksik yön;
  geçersiz Close ValueError ile mutasyondan önce reddedilir. Gerçek crossover
  okuyucusu iki yönde AL/SAT, ilk yön/değişmeyen yönde sinyal yok. Son ±1 gerçek
  pozisyon okuyucusunda mevcut 1 puanı ekler; caller/önceki sonuç korunur.
- **Kanıt:** 10 contract + 6 entegrasyon, önceki 2 runtime ile 18 hedef test
  green. Başlangıç RED ve kompleks/temporal fiyat edge düzeltmeleri kaydedildi;
  18 geçerli mutant assertion failure, error sıfır. Son birleşik 248 test dış
  socket engelli geçti (asyncio loopback serbest). Bağımsız reviewer 18 hedef
  testi yeniden çalıştırdı; Important/Critical/eyleme dönük minor bulgu yok.
  54 Python compile, 39 zorunlu belge/11 Boot bölümü, link/ADR/whitespace
  kontrolleri başarılı. Upstream validator ortamda yok; yapısal kontrol onun
  sonucu değildir. Config/dependency manifestleri aynı.
- **Kapsam/sınırlar:** RP-03 onaylı kapsamı kapandı; aynı prototype recurrence
  için matematik ve entegrasyon kanıtıdır, canonical OTT/kârlılık iddiası değil.
  Pozisyon okuyucusunun mevcut asgari veri sınırı değişmedi. Yeni API/UI/servis/
  dependency veya kalıcı migration yok; frontend/canlı sağlayıcı tekrar koşulmadı.
- **Git/sonraki adım:** Önceki değişiklikler korundu; HEAD `6ed1a59`, commit/push
  yok. ML yardımcılarının deterministik testleri; genel BL-001/M1/M2, dependency
  uyarıları ve operasyon/gizlilik/dağıtım kapıları açık. Plan/kanıt
  `docs/20-execution/2026-11-06-adr0006-implementation.md` içindedir.

### Önceki RP dar düzeltme kaydı — 2026-11-06

- **Yapılan:** RP-01 sıfır RSI/MACD okuyuculara girer, None/NaN eksik kalır.
  RP-02 GÜÇLÜ SAT mevcut satış boyutunu kullanır; ağırlık ve yüzdeler aynı.
  RP-04/05 private ATR/entry ve add/close doğrulaması, geçersiz girdide
  ValueError; indicator işi/saat/kayıt mutasyonundan önce. Gerçek scan sembol
  hatasını atlayıp devam eder; geçerli entry/R:R/P&L kayıt şekli korunur.
- **OTT kısmi:** Yalnız pandas API `.ffill()` düzeltildi; seed sırası ve alanlar
  aynı. `[100,110,90]` halen OTT `[100,100,100]`, sinyal `[NaN,sell,sell]`.
  Bu yanlış çıktı doğru beklenti olarak dondurulmadı. ADR-0006 Proposed hazır:
  loop önce seed, yeni nullable ilk gözlem/sonra ±1 `ott_trend`, mevcut okuyucular.
  Örnek `[100,105.995,96.72]` + `[NaN,+1,-1]` bağımsız hesapla doğrulandı.
- **Kanıt:** 8 alert testi önce 14 assertion failure, 10 DayTrader testi önce
  43 assertion/subtest failure, 2 runtime testi önce 4 TypeError; düzeltme sonrası
  green. Ana ajan 94 hedef ve birleşik 232 testi dış socket engelli geçti
  (asyncio loopback serbest). Bağımsız reviewer 66 yeni+korunmuş testi çalıştırdı;
  Important/Critical/eyleme dönük minor bulgu yok. İlgili plan:
  `docs/20-execution/2026-11-06-risk-position-fixes.md`.
  52 Python compile, 39 zorunlu belge/11 Boot bölümü, bağlantı/ADR/whitespace
  kontrolleri başarılı. Upstream CDSK validator ortamda yok; yapısal kontrol
  onun sonucu değildir. Config/dependency manifestleri değişmedi.
- **Karar/onay:** Genel “Sıradaki adım” mevcut hata düzeltmelerini yetkilendirir;
  henüz somutlaştırılmamış OTT veri alanı/başlangıcı için özel onay sayılmadı.
  AGENTS veri modeli kapısı gereği ADR-0006 açık onay bekler; seed/ott_trend
  kodda yok. Yeni servis/migration/version/dependency veya puan politikası yok.
- **Git/sınırlar:** Önceki değişiklikler korundu; HEAD `6ed1a59`, commit/push yok.
  Frontend/canlı sağlayıcı tekrar koşulmadı. Boyut/sermaye sınırları, sonsuz
  gösterge/diğer bozuk kayıt politikaları ve genel BL-001/M1/M2 kapanışı ayrı.
- **Sonraki adım:** Kullanıcı ADR-0006'yı onaylarsa yeni OTT sözleşmesini önce
  RED testlerle uygulamak; sonra gerçek sinyal/pozisyon zinciri ve kalan ML işleri.

### Önceki risk/pozisyon test kaydı — 2026-11-06

- **Yapılan:** `test_risk_analysis.py` (13), `test_position_recommendation.py`
  (16), `test_day_trader_risk.py` (17) eklendi. Risk toplama/eşik/volatilite,
  boyut/stop/hedef tutarları, pozisyon puanlama/karar/confidence, gerçek ATR/
  BUY-SELL-HOLD giriş/çıkış ve yerel miktar P&L/kayıt yaşam döngüsü kapsandı.
- **Kanıt:** 32 risk + 38 pozisyon + 34 DayTrader = 104 geçerli transient mutant
  assertion failure verdi, error yok; gerçek kodla hedef testler geri geçti.
  İki pencere mutantının ilk kaçışı asimetrik şok/yakın eşik fixture'larıyla
  kapatıldı. Bağımsız inceleme 46 hedef testi çalıştırdı ve sayı/formülleri
  doğruladı; Important/Critical/eyleme dönük minor bulgu yok.
  Son fixture'larla birleşik 212 test dış socket engelli geçti (asyncio loopback
  serbest). 49 Python compile, 39 zorunlu belge/11 Boot bölümü, bağlantı/ADR/
  whitespace ve üretim hesap/manifest değişmezliği kontrolleri başarılı.
- **Açık bulgular:** RP-01: RSI 0/MACD 0 truthiness ile atlanıyor. RP-02:
  puan -4/GÜÇLÜ SAT için %75–100 portföy etiketi; UI tüketimi kanıtlanmadı.
  RP-03: gerçek OTT mevcut pandas 3.0.1'de TypeError; ayrıca üretici/okuyucu
  `ott_signal`/`ott_trend` uyumsuzluğu. RP-04–05: doğrudan kısa/boş/geçersiz
  girdi sınırları. 13 bar BUY + NaN stop/hedef var; mevcut scan `<50` barı atlar.
  Bunlar düzeltildi/onaylandı sayılmaz ve golden regression beklentisi yapılmadı.
- **Sınırlar/kararlar:** Desteklenen hesaplama dilimi test edildi; tam risk/
  pozisyon entegrasyonu, gerçek emir/muhasebe veya olasılık kanıtı değil.
  Kural serileri ve DayTrader'ın iki consumer test girdisi gerçek hesaplama
  zincirlerinden ayrı işaretlendi. Üretim formülleri/config/API/model/manifest
  değişmedi; yeni ADR/commit/push yok. Önceki değişiklikler korundu; HEAD `6ed1a59`.
- **Sonraki adım:** RP-01–05 için somut düzeltme planı; mevcut hata düzeltmesi
  ile yeni finansal algoritma/girdi sözleşmesi/runtime kararı ayrılmalı. Ardından
  ML yardımcıları. Detaylar `docs/20-execution/2026-11-06-risk-position-tests.md`.
  Frontend/canlı sağlayıcı yeniden koşulmadı; genel BL-001/M1/M2 açık.

### Önceki BOS/kombinasyon test kaydı — 2026-11-06

- **Yapılan:** `backend/tests/test_bos.py` (15 test) ve
  `backend/tests/test_pattern_combos.py` (12 test) eklendi. Gerçek swing/kırılma
  ve iki yönlü FVG+OB/FVG+BOS zincirleri, ayrı matcher eşikleri, hacim/güç/
  tarih/fiyat cezaları, confidence sıralaması ve analyzer/girdi korunması kapsandı.
- **Kanıt:** 25 BOS + 15 geçerli kombinasyon mutantı assertion failure verdi,
  error yok; gerçek kodla 27 hedef test geçti. Birleşik 166 test dış socket
  engelli geçti (asyncio loopback serbest). Bağımsız incelemede Important/Critical
  yok; tarama sınırı metni düzeltildi: sondan 10. mum dahil, daha yeni 9 mum hariç.
  Son düzeltme sonrası 27 hedef test tekrar geçti; 46 Python compile, 39 belge/
  11 Boot bölümü, bağlantı/ADR/whitespace ve üretim değişmezliği kontrolleri temiz.
  Plan/kayıt `docs/20-execution/2026-11-06-bos-combo-tests.md` içindedir.
- **Mevcut sınırlar:** BOS batch'tir, swing komşularında geleceği kullanır;
  önceki ekstremumu seçer. Güç cap'inden sonraki hacim artışıyla 130 oluşabilir;
  combo confidence sezgiseldir. Bunlar causal/canlı strateji veya kârlılık kanıtı
  değildir. Matcher cache testleri ayrı gerçek dedektör kayıtlarıdır; tek OHLCV
  zinciri testleriyle karıştırılmaz. Nonpositive time_window politikası seçilmedi.
- **Karar/Git:** Üretim hesapları/config/API/veri modeli ve manifestler aynı;
  yeni ADR yok. Önceki değişiklikler korundu; aynı dal/HEAD `6ed1a59`, commit/
  push yok. Frontend/canlı sağlayıcı doğrulaması tekrar koşulmadı. Genel M1/M2,
  Constitution Draft, Next.js/Starlette/Streamlit ve CDSK validator sınırları açık.
- **Sonraki adım:** Risk/pozisyon hesaplarının deterministik testleri, ardından
  ML yardımcıları; kalan yapı/son-sinyal agregasyonu ve patern kenarları ayrıca açık.

### Önceki MACD/Bollinger test kaydı — 2026-11-06

- **Yapılan:** BL-001'in açık MACD/Bollinger değişken seri inceleme açığı için
  `backend/tests/test_technical_analysis.py` dosyasına 5 test eklendi. İki yönlü
  şok/reversion, sinyal/histogram sayıları ve warmup; Bollinger pencere/std/bant
  ile warmup sınırları doğrulandı. Girdi/index korunması da kapsandı.
- **Kanıt:** Bağımsız matematikten literal beklentiler; 12/12 bellek içi mutant
  assertion failure verdi. Gerçek kodla 17 hedef test yeniden green; birleşik
  139 test dış socket engelli geçti (Windows asyncio loopback serbest).
  Bağımsız inceleme sayıları ayrıca doğruladı; bulgu yok. 44 Python compile,
  39 zorunlu belge/11 Boot bölümü, bağlantı/ADR ve diff kontrolleri geçti.
  Plan/kayıt `docs/20-execution/2026-11-06-macd-bollinger-tests.md` içindedir.
- **Sınırlar/kararlar:** Üretim hesapları/config, API, veri modeli ve dependency
  manifestleri değişmedi; yeni ADR yok. Canlı Yahoo/RSS/SMTP ve frontend build/UI
  yeniden koşulmadı; önceki 10 cockpit senaryosu eski kanıttır. Next.js güvenlik/
  Starlette/Streamlit uyarıları ve upstream CDSK validator eksikliği sürüyor.
  BL-001/M1/M2 genel kapanışı iddia edilmez; Constitution hâlâ Draft.
- **Git/sonraki adım:** Önceki bekleyen değişiklikler korundu; aynı dal ve
  HEAD `6ed1a59`, commit/push yok. Sonraki alan BOS/kombolar, ardından risk/
  pozisyon ve ML yardımcıları; değişen finansal davranış/onay kapıları korunur.

### Önceki ADR-0005 uygulama kaydı — 2026-11-05

- **Onay/uygulama:** Somut ADR-0005 onay sorusuna “devam edelim” yanıtı kaydedildi;
  karar Accepted. RSS requests gövdesi byte olarak parse edilir; connect/read 8/8,
  2 ek deneme, 1.5/3.0 backoff ve yalnız onaylı retry nedenleri uygulanır.
  429/503 Retry-After varsa bu çağrı içinde tekrar ertelenir. Final HTTP hata
  gövdesi ve parser hatası retry/uydurma haber üretmez; response'lar kapanır.
- **Tanı/UI:** RSS/Yahoo `last_fetch_diagnostics` her fetch'te yenilenir;
  kaynak/durum/deneme/monotonic süre/hata kodu taşır. Yahoo `None` için neden
  uydurulmadan `no_data`; başarılı toparlanmada eski hata temizlenir. Haber
  özeti aynı fetch'in bağımsız kopyasını taşır. Legacy özet ve liste kendi
  kaynak durumlarını ve onaylı kısmi kesinti uyarısını ayrı gösterir.
- **Kanıt:** 22 RSS + 4 Yahoo + 5 legacy testi eklendi; birleşik 134 test dış
  socket engelli geçti (Windows asyncio loopback serbest). Hedefler: 46 RSS,
  18 provider/adaptör, 13 Streamlit AppTest. RED→GREEN ve response-close
  regresyonu kaydedildi. Bağımsız incelemede Important/Critical veya eyleme
  dönük minor bulgu yok. Ayrıntılar `docs/20-execution/2026-11-05-adr0005-implementation.md`.
- **Sınırlar:** Tanılar geçici instance belleğidir; merkezi metric/alarm,
  SDK içi HTTP sayısı, global rate limit veya toplam hard deadline değildir.
  History timeout SDK'nın tüm yardımcı isteklerini kapsamaz. Canlı Yahoo/RSS/
  SMTP, tam monolit browser ve frontend build/UI bu dilimde yeniden koşulmadı.
  Önceki Next.js güvenlik ve Starlette/Streamlit uyarıları açık; M1/M2 kapanmadı.
- **Git/belgeler:** `codex/m1-pattern-signal-tests`, HEAD `6ed1a59`; önceki
  değişiklikler korundu, commit/push yok. Manifest/endpoint/kalıcı şema değişmedi.
  44 Python compile, 39 zorunlu belge/11 Boot bölümü, yerel bağlantı/ADR ve
  whitespace kontrolleri geçti. CDSK upstream validator ortamda yok; yapısal
  kontrol onun sonucu değildir.
- **Sonraki adım:** BL-001 MACD/Bollinger için bağımsız beklenen değerli değişken
  seri testleri. Diğer M1, üretim, destek matrisi ve finansal davranış kararları
  kendi onay kapılarını korur.

### Önceki BL-003 hata-yolu kaydı — 2026-11-05

- **Yapılan:** BL-003 dar düzeltme/test dilimi: provider'ın tanımlı timeout'u
  `BISTDataFetcher` üzerinden `Ticker.history(timeout=...)` çağrısına iletilir;
  8/2/1.5 ayarları ve legacy override yokluğu değişmedi. Enjekte fetcher bu
  keyword'ü kabul etmelidir; repository çağıranları test edildi. Görünür integer
  HTTP >=400 RSS gövdesinden haber alınmaz; bozo/301 uyumu ve diğer kaynaklar korunur.
- **Kanıt:** 6 provider + 5 gerçek adaptör + 10 RSS kesinti testi eklendi;
  24 hedef ve birleşik 103 test dış socket engelli geçti (asyncio loopback serbest).
  Timeout zinciri 1 failure/3 error, HTTP gövdesi beş subtest red üretti;
  düzeltmelerden sonra green. 43 Python dosyası compile ve diff kontrolü geçti.
  Bağımsız incelemede Important/Critical bulgu yok; Proposed ADR enum notu düzeltildi.
- **Sınırlar:** History timeout toplam işlem/SDK deadline'ı değildir. Kurulu
  yfinance 1.8.0 lazy timezone isteğinde kendi timeout'unu kullanır. RSS hâlâ
  URL parser'ı ile çalışır; açık request timeout/retry ve kaynak metriği yok.
  Feedparser redirect metadata'sı nihai hatayı örtebilir; yalnız bilinen hata
  guard'ı doğrulandı. Frontend build/UI ve canlı Yahoo/RSS/SMTP yeniden koşulmadı.
- **Yeni karar:** `ADR-0005` Proposed; RSS connect/read 8/8, en fazla 2 ek deneme,
  1.5/3.0 bekleme, retry sınıflandırması/Retry-After erteleme ve geçici kaynak
  tanı/metin sözleşmesi hazır. Açık onay bekler; mevcut dar düzeltme bu yeni
  politikanın onayı veya uygulaması değildir. BL-003/M1 açık.
- **Belgeler/Git:** INTEGRATIONS/OBSERVABILITY/API_CONVENTIONS mevcut/hedef
  ifadeleri uzlaştırıldı. `codex/m1-pattern-signal-tests`, HEAD `6ed1a59`;
  önceki değişiklikler korundu, commit/push yok, dependency manifestleri değişmedi.
  Upstream CDSK validator ortamda yok; yapı/bağlantı kontrolü ayrıca kaydedilir.
- **Sonraki adım:** Kullanıcı ADR-0005'i onaylarsa yeni RSS politikası ve tanı
  okuyucularını testlerle uygulamak. Diğer M1/ML/üretim onay kapıları korunur.

### Önceki BL-002 uygulama kaydı — 2026-11-05

- **Onay/uygulama:** Kullanıcı A metinleri ve ADR-0004 için sorulan doğrudan
  soruya “devam et” yanıtını verdi. İkisi Accepted olarak kaydedildi; başka
  servis, migration, deployment veya finansal hesap politikası onayı sayılmaz.
- **Yapılan:** Cockpit metadata'yı korur; sağlayıcı/demo/bilinmeyen ayrımı ve
  genel uyarı görünür. Analiz hata/sembol/oturum değişiminde temizlenir, geç gelen
  eski yanıt sonucu geri getirmez. RSI başlangıç/boş/veri yok/hata metinleri ayrıldı.
  RSS yayın/çekilme zamanı ayrılır; NLP hatası/nonfinite `None/failed`, gerçek
  sıfır nötrdür. Legacy haber nullable alanları tüketir; ML deneysel, uyumu
  sezgiseldir. Sabit olasılıklar kaldırıldı; geçersiz yayılımda fallback senaryo,
  bant, uyum ve ilgili risk kartı gösterilmez. Geçerli formüller değişmedi.
- **Kanıt:** 82 test dış socket bağlantıları engelliyken geçti (14 RSS/NLP,
  8 kontrollü Streamlit AppTest dahil; Windows asyncio için loopback serbest).
  10/10 gerçek üretim cockpit senaryosu, son Next.js build ve iki Node sözdizimi
  kontrolü geçti. 41 Python dosyası compile kontrolü başarılı. Yeni testler önce
  eski davranışta başarısız oldu; geçersiz uyum ve bozuk metadata düzeltmeleri
  ayrıca red→green doğrulandı. Bağımsız incelemede önemli/engelleyici bulgu yok.
- **Sınırlar:** AppTest gerçek fonksiyonları AST ile yükleyip kontrollü IO ile
  render eder; tam monolit browser veya canlı model/sağlayıcı kanıtı değildir.
  Canlı Yahoo/RSS/SMTP koşulmadı. Yerel Streamlit/Plotly/NLP kurulumu mevcut
  requirements içindir; manifest/sürüm politikası değişmedi. Next.js güvenlik
  riski, Starlette/Streamlit deprecation uyarıları ve upstream CDSK validator
  eksikliği açık. ML ufuk/tarih, kesin eğitim penceresi/kalibrasyon ve eşik uyumu
  bu dilimde değiştirilmedi; genel BL-002/M1/M2 kapanışı iddia edilmez.
- **Git/sonraki adım:** `codex/m1-pattern-signal-tests`, HEAD `6ed1a59`;
  önceki değişiklikler korundu, commit/push yok. Sonraki iş BL-003 mevcut
  entegrasyon hata yolları; yeni politika/servis kararı onaysız alınmaz.
  Ayrıntılar `docs/20-execution/2026-11-05-bl002-implementation.md` içinde.

### Önceki BL-002 inceleme ve öneri kaydı — 2026-11-05

- **Yapılan:** BL-002 için cockpit, legacy haber/ML, API/provider ve sentiment
  kaynakları incelendi. Kesin metinler ve A uygulama dilimi
  `docs/00-product/BL002_DATA_DISCLOSURE.md` içinde Proposed. RSS tarih ve duygu
  hata ayrımı için B sözleşmesi `ADR-0004` Proposed hazırlandı; henüz Accepted değil.
- **Bulgular:** Analiz UI'sı metadata'yı kaybediyor ve hata halinde eski sonucu
  temizlemiyor. Legacy haber gerçek RSS kullanıyor; mock yardımcılar için diğer
  Python modüllerinde doğrudan çağrı bulunmadı. ML uyum skoru olasılık değildir;
  grafik interpolasyonu/bandı istatistiksel güven aralığı olarak doğrulanmadı.
  RSS eksik/geçersiz yayın tarihi bugünün saatiyle dolduruluyor; NLP hatası `0.0`
  dönerek nötr gözlemle birleşiyor. Bunlar uygulanmış düzeltme değildir.
- **İnceleme:** Bağımsız inceleme sabit %30/%40/%30 ML senaryo olasılıklarını
  ek bulgu olarak verdi; M-09–11 metin/guard önerileri eklendi. Model ufku/tarih
  ve sentiment etiket/filtre eşik tutarsızlıkları kalan iş olarak kaydedildi.
- **Kapsam:** Bu adım kaynak/AST incelemesi ve belge hazırlığıdır; üretim kodu,
  test, dependency veya kalıcı veri değişmedi. Tarayıcı/Streamlit/canlı servis,
  backend testleri ve build bu belge adımında yeniden çalıştırılmadı.
- **Belge doğrulaması:** Öneri/ADR bağlantıları ve Proposed durumları, 39 zorunlu
  belge/11 Boot bölümü, 39 Python dosyası sözdizimi ve whitespace kontrolü geçti.
  Upstream CDSK validator ortamda yok; yapısal kontrol onun sonucu sayılmaz.
- **Onay:** BACKLOG BL-002 ürün metni kapısı ve AGENTS veri modeli kapısı gereği
  A metinleri/davranışları ile ADR-0004 açık onay bekliyor. Genel “devam” talebi
  bu somut metin/sözleşmelerin özel onayı diye kaydedilmedi.
- **Git/sonraki adım:** `codex/m1-pattern-signal-tests`, HEAD `6ed1a59`;
  önceki değişiklikler korundu, commit/push yok. Onaydan sonra uygulama planı,
  ilgili testler ve düzeltmeler; BL-002/M1 bu aşamada kapanmaz.

### Önceki deterministik tarayıcı doğrulaması — 2026-11-05

- **Yapılan:** BACKLOG sıralı iş 3 tamamlandı: `--deterministic` test modu,
  `scripts/frontend_ui_scenarios.mjs`, Makefile hedefi ve README yönergeleri.
  Gerçek Next.js ekranında test içi API yanıtları; auth/watchlist/analiz/paper/RSI
  başarı, boş/kısmi/veri yokluğu, açık/kapalı RSI tercih kalıcılığı ve 503 sonrası
  temizleme/yeniden deneme sınanır. Üretim UI/API veya lockfile değişmedi.
- **Kanıt:** 5/5 senaryo hem `next dev` hem `next build` sonrası `next start`
  ekranında headless Edge/Playwright 1.62.1 ile geçti. Beklenmeyen dış istek yok;
  backend, Yahoo/RSS/SMTP başlatılmadı. Next.js üretim derlemesi, iki Node scripti
  sözdizimi kontrolü geçti. Test tarayıcısında tercih kaydı devre dışı bırakılınca
  kalıcılık senaryosu hatayı yakaladı; Playwright yokluğunda exit 1 doğrulandı.
- **İnceleme:** Bağımsız incelemede engelleyici bulgu yok. Yeniden denemede
  başarı durumunun geri gelmesi de sınanıyor; analiz fixture'ı yalnız UI'nın
  tükettiği alanları kapsar, tam API sözleşmesi değildir.
- **Sınırlar:** Canlı UI smoke ve 60 backend testi bu frontend adımında yeniden
  koşulmadı; önceki backend kanıtı aşağıda korunur. Makefile'ın yeni hedefi mevcut
  CI'ye otomatik eklenmedi. Tam CDSK validator hâlâ yok; Next.js kurulum güvenlik
  uyarısı açık. Boş sonuç metni ve iç içe API hata ayrıntısı sunumu backlog'da açık.
- **Git/kararlar:** `codex/m1-pattern-signal-tests`, HEAD `6ed1a59`; önceki
  değişiklikler korundu. Commit/push yapılmadı. Yeni ADR yok; M1/M2 tamamlanmadı.
- **Sonraki adım:** BACKLOG sıralı iş 4 / BL-002: cockpit ve legacy sentiment/ML
  ekranlarında gerçek/mock/deneysel niteliği, kaynak ve zaman görünürlüğünü
  incelemek; gereken ürün metnini somutlaştırmak. Ürün metni onayı ve mevcut
  mimari/güvenlik/servis/migration onay kapıları korunur.

### Önceki patern/sinyal doğrulaması — 2026-11-05

- **Devam:** Kullanıcı projeyi inceleyip kaldığımız yerden devam etmemizi istedi;
  önceki bekleme kaldırıldı. Başlangıç Git durumu temiz, HEAD `6ed1a59` idi.
  Çalışma dalı `codex/m1-pattern-signal-tests`; bu adımda commit/push yapılmadı.
- **Yapılan:** BL-001 / sıralı iş 2 için 14 patern ve 11 sinyal testi eklendi.
  Doji/gölge/saran/yıldız, çift tepe/dip, FVG eşik/dolum, order block yön/güç;
  RSI, MACD/EMA kesişimleri, SuperTrend/OTT, Bollinger, hacim ve çoğunluk
  kararı sentetik verilerle sınandı. Üretim kodu, şema ve requirements değişmedi.
- **Kanıt:** 60 backend testi geçti; aynı paket dış socket bağlantıları
  engellenerek de geçti (Windows asyncio için loopback serbest). Dört bellek içi
  patern/sinyal mutasyonu assertion hatasıyla yakalandı. 39 Python dosyası
  sözdizimi, 39 zorunlu CDSK belge ve 11 Boot bölümü kontrolü geçti. Upstream
  CDSK validator bu makinede bulunamadı; frontend build/canlı UI smoke koşulmadı.
- **Ortam:** Yerel `.venv`, uygulamayla gelen Python 3.12.14'ün paketlerini
  kullanır; eksik test bağımlılıkları yalnız bu ortama kuruldu. Bu taşınabilir
  bir dependency lock veya destek matrisi onayı değildir. Starlette/httpx
  deprecation uyarısı var; önceki LibreSSL uyarısı bu ortamda görülmedi.
- **Sınır:** Bu testler mevcut dar hesaplamayı doğrular; paternlerin finansal
  geçerliliği/öngörü başarısı kanıtlanmış değildir. BOS/kombolar, risk/pozisyon
  yardımcıları, ML ve değişken seri MACD/Bollinger kapsamı açık; BL-001/M1 kapanmadı.
- **İnceleme:** Bağımsız incelemede engelleyici bulgu yok; iki yeni modül
  ayrıca geçti. Üç düşük öncelikli sınır testi BACKLOG'da açık bırakıldı:
  kısa gölge tam eşitliği, yıldız ilk gap'i ve çift tepe/dip %2 sınırı.
- **Karar:** Yeni ADR gerekmiyor. Mevcut güvenlik, migration, vendor/maliyet,
  üretim operasyonu ve ürün metni onay kapıları korunuyor.
- **Sonraki adım:** BACKLOG sıralı iş 3: kontrollü API yanıtlarıyla başarı,
  boş sonuç, veri yokluğu ve RSI tercih kalıcılığını koşulsuz tarayıcı testlerine
  almak. Üretim koduna mock modu eklenmez.

### Önceki bekleme ve Git teslim kaydı — 2026-11-05

- **Bekleme talebi:** Kullanıcı "şimdilik kalsın" dedi; yeni uygulama/test
  geliştirmesi yapılmadan mevcut CDSK durumu ve kalan planlar yazılı bırakıldı.
  M1/M2/M3 kapanış ölçütleri ve onay kapıları
  `docs/20-execution/BACKLOG.md` içinde toplandı. Devamda önce bu kayıt okunur.
- **Git kaynak durumu:** `origin` fetch/push:
  `https://github.com/celebigilfatih/trader.git`; yerel dal `main`.
  Önceki bekleme kontrolünde CDSK belgeleri ve yeni uygulama dosyalarının çoğu
  untracked idi. Kullanıcı sonrasında açıkça commit + push istedi; bu teslim
  değişiklikleri ve kalan iş planını birlikte sürümlemeyi amaçlar.
  Push öncesi uzak `main`, yerel başlangıç commit'i `2ca5644` ile eşleşti.
  Push sonucu uzak ref ile ayrıca doğrulanmalıdır; uygulama geliştirmesine
  verilen ara devam eder, bu işlem üretime dağıtım değildir.
- **Git teslim doğrulaması:** 35 backend testi, CDSK validator, UI smoke scripti
  sözdizimi ve Next.js üretim derlemesi yeniden geçti. Kurulumdaki Next.js
  güvenlik uyarısı ve Python/LibreSSL uyarısı açık kaldı; canlı UI smoke
  yeniden koşulmadı. Yerel dependency/build klasörleri commit kapsamı dışında.

- **Yapılan:** BL-001 için gerçek teknik hesaplamaları sınayan 12 test ve gerçek
  hesaplayıcılı 3 API testi eklendi. Üretim kodu, bağımlılık, şema veya güvenlik
  politikası değiştirilmedi. Kullanıcının mevcut değişiklikleri korundu.
- **Kanıt:** 35 backend testi ve CDSK validator geçti. Frontend build/UI smoke
  bu adımda yeniden koşulmadı. Python 3.9/LibreSSL uyarısı sürüyor.
  Aynı 35 test socket bağlantıları engellenerek de geçti; bellekte uygulanan
  hatalı EMA ve sabit RSI varyantlarını yeni testlerin yakaladığı doğrulandı.
  Genel diff kontrolünde önceden var olan `README.md:215` son boşluk uyarısı
  görüldü; bu oturumun test ve ilgili mimari/kalite/backlog dosyalarında
  whitespace kontrolü geçti.
- **Tutarlılık:** M1, ROADMAP ile aynı şekilde In Progress olarak düzeltildi;
  mimari ve kalite belgelerinde mevcut API/cockpit ile eski hedef ifadeleri ayrıldı.
- **Eksikler:** Patern/sinyal/ML test kapsamı, deterministik tarayıcı hata
  senaryoları, mock/deneysel etiketlerin tüm ekranlarda görünürlüğü ve gerçek
  timeout iletimi. M2 alarm/snapshot ve M3 üretim işleri tamamlanmadı.
  Bağımsız test/belge incelemesi engelleyici bulgu üretmedi; MACD/Bollinger için
  değişken seri fixture'ları düşük öncelikli ek kapsam olarak BL-001'de açık.
- **Onay sınırı:** Yeni ADR gerektiren karar alınmadı. Constitution Draft;
  veri/lisans, şema/migration, üretim auth, vendor/maliyet ve operasyon
  değişikliklerinde açık onay gerekliliği korunuyor.
- **Sonraki adım:** `docs/20-execution/BACKLOG.md` içindeki sıralı beş işin
  ikinci maddesi: gerçek patern ve sinyal hesaplarının ağsız olumlu/olumsuz
  senaryolarını eklemek. İlk madde dar hesaplama test adımı olarak tamamlandı.

### Önceki oturumların tarihsel kayıtları (güncel test sayısı yukarıdadır)

- **Session summary:** Mevcut Streamlit/BIST prototipi incelendi ve CDSK 0.1.0
  mevcut README/koda dokunmadan yönetim katmanı olarak uygulanmıştı; bu oturumda
  FastAPI API çekirdeği (`backend/app`) ve Next.js MVP cockpit’i (`frontend/app`)
  birlikte işletilebilir hale getirildi.
- **Documents updated:** CDSK iskeleti; ürün, mimari, execution, kalite,
  operasyon, AI, README, changelog ve ignore politikası.
  Frontend başlangıç talimatları eklendi.
  Ayrıca backend API eklemesi de izlenebilir biçimde kayıtlandı.
  Frontend build (`npm run build`) doğrulaması ve tek komutlu
  geliştirme başlatma scripti (`run_dev.py`) eklendi.
- **Tests and verification:** `backend/tests/test_contract.py` ve
  `backend/tests/test_services.py` ile birlikte API ve servis kontratları test edildi;
  unittest ile toplam **20 test** geçti.
  Frontend tarafı için `npm install` ve `npm run build` ile derleme kontrolü tamamlandı.
  `run_dev.py` ile tek komutlu başlatma ve `run_dev.bat` script doğrulaması da eklendi.
  `run_dev.py` şimdi hazır olma zaman aşımı ve kapanış davranışıyla daha stabil
  şekilde yönetiliyor; süreçlerden biri erken kapanırsa diğeri de güvenli şekilde
  kapatılıyor.
  Operasyonları standartlaştırmak için `Makefile` eklendi (`make run-dev`, `make test`,
  `make smoke`, `make ci` hedefleriyle hızlı tekrarlayan akışlar destekleniyor).
  Ayrıca `scripts/run_dev_smoke.py` ile kısa süreli entegrasyon smoke kontrolü
  genişletildi; sağlık kontrolüne ek olarak auth invite/magic, watchlist ve
  portfolio kontrat doğrulamaları ile frontend ana ekran kontratı da koşuyor.
  Bu doğrulama seti `make ci` içinde otomatik olarak koşuyor.
-  `make smoke-ui-stack` hedefi eklendi; backend ve frontend’i tek komutla
  başlatıp Playwright tabanlı UI smoke’u çalıştıran otomasyon eklendi.
-  `make smoke-all` hedefi eklendi; `make smoke` ve `make smoke-ui-stack` ardışık
   koşarak API ve browser tabanlı UI doğrulamalarını tek bir akışta birleştirir.
-  `make ci-e2e` ve `make ci-full` hedefleri eklendi; biri CI temelleri üstüne
  (`ci-core`) tarayıcı e2e akışını ekler (`smoke-ui-stack`), diğeri de tek satırda
  bu kapanışı sağlar.
-  Günlük doğrulama playbook’u (`smoke`, `smoke-all`, `ci-e2e`, `ci-full`) standartlaştırıldı.
-  `make ci-core` hedefiyle `make ci` akışındaki `smoke` adımı ayrıştırılarak
  `make ci-e2e` içinde port çakışmasına yol açan ardışık `smoke` koşusundan
  kaçınıldı.
-  Günlük kapanış akışını tek adımda çalıştırmak için `make qa` hedefi eklendi
  (`make ci-e2e` alias’ı).
-  `run_dev.py`, frontend’i başlatırken `NEXT_PUBLIC_API_BASE` env değişkenini
  enjekte etmeye başladı; `frontend/app/page.js` ise varsa bu değeri temel API
  URL’i olarak kullanıyor.
-  `backend/app` modüllerindeki yanlış paket importları düzeltildi, böylece
  `make ci` altındaki `run_dev` smoke adımı stabil çalışır hâle geldi.
-  `MarketDataProvider` üretim kodu testlerde `fetcher` enjeksiyonu destekleyecek
  şekilde genişletildi; böylece ağ dışı koşullarda deterministik tekrar edilebilirlik sağlandı.
-  `backend/tests/test_market_data.py` ile retry ve veri yokluğu senaryoları test
   edildi; `backend/tests/test_api_contracts.py` ile auth gerektiren ana endpoint
   akışları kontrat odaklı doğrulandı.
-  `run_dev smoke` akışında frontend kontrat doğrulaması, Next.js chunk URL keşfi ve
  bölüm metinlerinin tespiti için daha dayanıklı hale getirildi.
-  `scripts/run_frontend_ui_smoke.mjs`, Playwright modülünü `frontend/node_modules/playwright`
  fallback ile yükleyebilir hale getirildi; böylece `make smoke-ui` yalnızca kök düzeyinde
  kurulu olmasa bile çalışır.
-  `frontend/app/page.js` içindeki temel aksiyon alanlarına (`watchlist`/`portfolio`/`logout`)
  test kimlikleri (`data-testid`) eklendi; bu sayede `scripts/run_frontend_ui_smoke.mjs`
  daha az kırılgan hale getirildi.
-  RSI screener sonuçlarının tablo görünümünde veri kalitesi için kritik meta alanları
  (`source`, `fetched_at`, `as_of`, `staleness_seconds`) gösterilmeye başladı.
-  `scripts/run_frontend_ui_smoke.mjs` RSI tarama akışını doğruluyor; yeni eklenen
   metadata sütunlarının UI başlıkları beklenir şekilde bulunuyor.
-  RSI screener artık tek bir sembolde veri yokluğunda akışı durdurmak yerine o
   sembolü atlayıp kalan valid sonuçları geri döndürüyor; böylece watcher listesi
   içinde bulunabilecek tarihsel/temizlik problemi olan semboller kullanıcıyı durdurmuyor.
-  RSI screener yanıtında erişilemeyen semboller `skipped_symbols` alanında döndürülüyor;
   frontend kartında bu durum kullanıcıya uyarı olarak gösteriliyor.
-  RSI screener ayrıca `warnings` alanı ile atlanan semboller için gerekçe (`symbol` +
   `code` + `reason`) üretmeye başladı; UI kartında her sebep satır satır gösteriliyor.
-  RSI uyarı paneli, atlanan sembol sayısı + kod bazlı özet etiketlerini
   tek bakışta göstermeye ve detayları açıp kapatmaya olanak tanıyacak şekilde
   güncellendi; bu sayede kullanıcı akışı daha okunaklı oldu.
-  UI smoke senaryosu, RSI paneli uyarısı varsa detay açma etkileşimini doğrulayacak
   şekilde güvenli/koşullu kontrole alındı.
-  RSI `warnings` kontratında `insufficient_ohlcv_for_rsi` senaryosu backend testleriyle
   doğrulandı.
- **Decisions recorded:** CDSK benimsenmesi; API-first mimari hedefi ve pilot
  uçtan uca pilot akışlar kabul edildi; 3D profil N/A.
- **Remaining risk:** Runtime test bağımlılıkları, gerçek/veri doğruluk etiketleme
  politikası onayı, operasyon sahipliği kararı ve Next.js/React güvenlik
  güncelleme planı.
- **Historical next step:** Çevrimdışı test altyapısı kuruldu; kalan kapsam için
  yukarıdaki güncel devir ve execution backlog izlenmelidir.

> Kısa tut. Tam sohbet özeti yazma; bir sonraki oturumun güvenle devam etmesi
> için gerekli kalıcı bağlamı yaz.

## Revision History

| Version | Date | Summary |
|---|---|---|
| 1.48 | 2026-11-06 | Simple ML-04–06 kapandı;13 yeni test,83 ML/331 backend testi ve11 mutant. Model/config/dependency aynı; sonraki ML-07/08/takvim kararı, genel M1 açık. |
| 1.47 | 2026-11-06 | ADR-0007 açık onayla uygulandı;30 yeni/70 ML testi ve318 backend testi. ML-01–03/advanced ML-04 kapandı; Simple ve kalan politika/M1 açık. |
| 1.46 | 2026-11-06 | ML-01–03 tasarımı ve ADR-0007 Proposed; purge/train-only bounds/ortak ölçek/state/artifact sözleşmesi açık onay bekliyor. 40 mevcut ML testi geçti; üretim değişmedi. |
| 1.45 | 2026-11-06 | 40 ML helper/feature/eğitim testi ve 80 kontrollü mutant; 288 backend testi. Üretim ML değişmeden ML-01–08 açık bulguları kaydedildi; sonraki iş hedef/ölçek/ön işleme düzeltme planı. |
| 1.44 | 2026-11-06 | ADR-0006 açık onayla uygulandı; seed/yön ve gerçek okuyucu zinciri, 16 yeni test/248 backend testi; RP-03 onaylı kapsamı kapandı, genel M1 açık. |
| 1.43 | 2026-11-06 | RP-01/02/04/05 ve OTT runtime dar düzeltmesi, 20 RED→GREEN regresyon/232 test; tam OTT seed/yön sözleşmesi ADR-0006 Proposed onay bekler. |
| 1.42 | 2026-11-06 | Risk/pozisyon/DayTrader için 46 test, 104 geçerli mutant, 212 backend testi; mevcut RP-01–05 kusurları ayrı açık kaydedildi, finansal davranış değişmedi. |
| 1.41 | 2026-11-06 | Temel BOS/kombolar için 27 yeni test, 40 mutant ve toplam 166 backend testi; bağımsız inceleme ve tarama sınırı metin düzeltmesi; risk/ML ve M1 açık. |
| 1.40 | 2026-11-06 | BL-001 MACD/Bollinger değişken seri açığı 5 yeni test ve 12 mutantla kapatıldı; 139 backend testi; finansal davranış değişmedi, genel M1 açık. |
| 1.39 | 2026-11-05 | ADR-0005 onaylanıp RSS bütçe/tanı ve legacy kaynak uyarısı uygulandı; 31 yeni testle 134 backend testi, bağımsız inceleme tamamlandı; üretim/M1 kapıları açık. |
| 1.38 | 2026-11-05 | BL-003 timeout iletimi/bilinen RSS HTTP hata guard'ı, 21 yeni testle 103 backend testi; ADR-0005 yeni RSS bütçe/tanı politikası Proposed. |
| 1.37 | 2026-11-05 | A metinleri ve ADR-0004 onaylanıp uygulandı; 82 test/10 UI senaryosu, üretim build ve bağımsız inceleme tamamlandı; mevcut hesap/altyapı riskleri ayrı bırakıldı. |
| 1.36 | 2026-11-05 | BL-002 kaynak incelemesi ve kesin ekran metinleri Proposed; RSS/NLP doğruluk sözleşmesi için ADR-0004 onay bekliyor, çalışma zamanı değişmedi. |
| 1.35 | 2026-11-05 | Beş deterministik UI senaryosu dev/üretim ekranlarında doğrulandı; ayrı komut ve test fixture'ları, canlı smoke bekleyici düzeltmesi, ürün metni/hata sunumu sınırları kaydedildi. |
| 1.34 | 2026-11-05 | Geliştirmeye devam edildi; 25 patern/sinyal testiyle toplam 60 backend testi doğrulandı, Windows test ortamı ve CDSK validator sınırı kaydedildi; sıradaki iş deterministik tarayıcı senaryoları. |
| 1.33 | 2026-11-05 | Kullanıcı isteğiyle geliştirmeye ara verildi; CDSK/M1–M3 kalan işleri, onay kapıları ve yerel Git remote/sürümlenme durumu yazılı bırakıldı. |
| 1.32 | 2026-11-05 | Gerçek hesaplayıcı için 15 test eklendi (toplam 35); M1 durum, mimari/test belgeleri ve sıralı backlog güncellendi. |
| 0.1 | 2026-08-17 | CDSK proje şablonu oluşturuldu |
| 0.2 | 2026-08-17 | Current-state ürün, mimari, riskler ve handoff dolduruldu |
| 0.3 | 2026-08-19 | ADR’lar ve hedef ürün/ mimari kararı alındı; roadmap ve execution
güncellendi |
| 0.4 | 2026-08-19 | FastAPI API çekirdeği ile Next.js MVP cockpit’i operasyona
   hazırlandı; frontend başlatma/derleme doğrulaması ve dokümantasyon güncellendi |
| 0.5 | 2026-08-19 | Tek komutlu geliştirme akışını geliştirmek için `run_dev.py` ve
	`run_dev.bat` eklendi; `--install-frontend-deps` seçeneği ile başlatma kolaylaştı |
| 0.6 | 2026-08-19 | `run_dev.py` hazır olma ve kapanış davranışı güçlendirildi; `--ready-timeout`,
  `--check-interval`, `--shutdown-timeout` seçenekleri eklendi; erken sonlanan süreç
  için eşzamanlı kapatma davranışı tanımlandı. |
| 0.7 | 2026-08-19 | `scripts/run_dev_smoke.py` eklendi; `run_dev` akışını kısa süreli
  otomatik sağlık + kapanış smoke ile doğrulayacak geliştirme aracı getirildi. |
| 0.8 | 2026-08-19 | Makefile eklendi (`run-dev`, `test`, `build-frontend`, `smoke`, `ci`
  hedefleriyle standart geliştirme/CI başlatma ve doğrulama akışı sağlandı). |
| 0.9 | 2026-08-19 | GitHub Actions CI (`.github/workflows/ci.yml`) eklendi; `make ci`
  hedefindeki backend import hataları düzeltildi ve smoke doğrulaması güvenli şekilde
  geçti. |
| 1.0 | 2026-08-19 | Offline deterministik market-data retry ve kontrat testleri eklendi;
  API contract testleri ile ana uç noktalar doğrulandı; backend test kapsamı
  18'e çıkarıldı ve `make ci` altında geçti. |
| 1.1 | 2026-08-19 | API kontrat kapsamı genişletildi; auth eksikliği ve input
  doğrulama senaryoları (401/400/422) kontrat testleri ile 18 testlik
  backend setine eklendi. |
| 1.2 | 2026-08-19 | `run_dev_smoke.py` kapsamı genişletildi; smoke içinde auth
  invite/magic, watchlist ve portfolio kontrat doğrulamaları eklendi. |
| 1.3 | 2026-08-19 | Frontend kontrat kontrolü eklendi; smoke akışı artık
  Next.js kök sayfasında beklenen HTML marker'larını doğrulayarak frontend
  ayağa kalkışını doğruluyor. |
| 1.4 | 2026-08-19 | `run_dev.py` ile başlatılan frontend şimdi dinamik API base
  (`NEXT_PUBLIC_API_BASE`) alıyor; `frontend/app/page.js` bu değeri öncelikli
  olarak kullanarak çalıştığı ortamla uyumu artırıyor. |
| 1.5 | 2026-08-19 | Frontend smoke kontratı daha dayanıklı hale getirildi; app/page
  chunk URL çıkarımı esnekleştirildi, `/_next/static/chunks/app/page.js` fallback
  eklendi ve bölüm metinlerinin kontrolü için ek doğrulama eklendi. |
| 1.6 | 2026-08-19 | Smoke akışı kritik iş akışlarına genişletildi; auth
  sonrası watchlist ve portfolio için add/remove senaryoları doğrudan doğrulanıyor.
  |
| 1.7 | 2026-08-19 | Smoke kontrat kapsamı derinleştirildi; watchlist set/update
  akışı ve portföy yeniden ekleme (weighted average) kontrolüyle upsert davranışı
  doğrulandı. |
| 1.8 | 2026-08-19 | `make smoke-ui` hedefi ve `scripts/run_frontend_ui_smoke.mjs`
  ile Playwright tabanlı opsiyonel UI e2e smoke hattı eklendi; oturum sonrası
  watchlist/portfolio etkileşimleri doğrulanabiliyor. |
| 1.9 | 2026-08-19 | UI e2e smoke senaryosu için frontend bileşenlerine `data-testid`
   eklenerek testlerin selector kırılganlığı azaltıldı; `README` ve `CHANGELOG`
   ilgili yönergelerle güncellendi. |
| 1.10 | 2026-08-19 | UI smoke scripti, Playwright’i doğrudan frontend bağımlılığı
  fallback’ından da okuyacak şekilde güncellendi; `make smoke-ui` gerçek uçtan uca
  ortamda doğrulandı. |
| 1.11 | 2026-08-19 | `scripts/run_frontend_ui_smoke_with_stack.py` ve
  `make smoke-ui-stack` eklendi; stack otomatik başlatma + Playwright UI e2e smoke
  tek komut akışı tamamlandı. |
| 1.12 | 2026-08-19 | `make smoke-all` hedefi eklendi; `smoke` ve `smoke-ui-stack`
  ardışık çalışarak tam kapsamlı doğrulamayı tek adımda tetikliyor. |
| 1.13 | 2026-08-19 | `make ci-e2e` ve `make ci-full` hedefleri eklendi; `ci` akışı
   üzerine Playwright e2e doğrulama katmanı eklenmiş oldu. |
| 1.14 | 2026-08-19 | `README` içinde `smoke`/`smoke-all`/`ci-e2e`/`ci-full`
  hedeflerine göre yerel doğrulama playbook’u standardize edildi. |
| 1.15 | 2026-08-19 | `make ci-e2e` akışı port güvenliği için yeniden düzenlendi;
  `ci-core` hedefi eklendi ve `ci-e2e` artık doğrudan `smoke-ui-stack` üzerinden
  Playwright e2e katmanı ekliyor. |
| 1.16 | 2026-08-19 | `make qa` hedefi eklendi; lokal kapanış QA akışı için tek komut
  aliası sağlandı (`ci-e2e`). |
| 1.17 | 2026-08-19 | README ile dokümantasyondaki lisans referansını destekleyecek
  olarak `LICENSE` dosyası eklendi; kalıcı lisans kaynağı belirsizliği giderildi. |
| 1.18 | 2026-08-19 | RSI screener çıktısına veri kaynağı ve zaman damgası meta alanları
  (`source`, `fetched_at`, `as_of`, `staleness_seconds`) eklendi; kontrat testiyle
  ekran düzeyinde doğrulandı. |
| 1.19 | 2026-08-19 | RSI screener tablosu, source/fetched_at/as_of/staleness bilgilerini
  kullanıcı arayüzünde göstermeye başladı; frontend build doğrulamasıyla uyumlu hale getirildi. |
| 1.20 | 2026-08-19 | Frontend UI smoke senaryosu `RSI tarama` akışını ve
  meta kolon başlıklarını doğrulayacak biçimde genişletildi (`rsi-screen-button`,
  `rsi-results-table` test kimlikleri ile). |
| 1.21 | 2026-08-19 | RSI screener endpointi, tek bir sembol için veri yoksa 500 üretmek
  yerine o sembolü atlayıp kalan symbol sonuçlarını döndürecek şekilde dayanıklı
  hale getirildi; test kapsamı buna göre genişletildi. |
| 1.22 | 2026-08-19 | RSI screener yanıtına `skipped_symbols` eklendi; erişilemeyen
  semboller frontend RSI kartında uyarı olarak gösterilecek şekilde UI + kontrat
  uyumu tamamlandı. |
| 1.23 | 2026-08-19 | RSI screener `warnings` alanını ekleyerek atlanan sembol nedenlerini
  kontrat ve UI seviyesinde raporladı; smoke senaryosu bu alanı da doğrular hâle getirildi. |
| 1.24 | 2026-08-19 | RSI screener `warnings` için standart hata kodları eklendi
  (`market_data_unavailable`, `insufficient_ohlcv_for_rsi`) ve kontrat+UI uyumu
  bu şekilde stabilize edildi. |
| 1.25 | 2026-08-19 | RSI `warnings` görüntüleme, kod korumalı ama kullanıcı dostu
  Türkçe etiketli biçime çevrildi; ön yüzde teknik nedenler okunabilir hale getirildi. |
| 1.26 | 2026-08-19 | RSI uyarıları kullanıcı arayüzünde ikonlu ve renk tonlu uyarı
  kartlarıyla görselleştirildi; teknik kodlar korunurken okunabilirlik arttı. |
| 1.27 | 2026-08-19 | RSI uyarı kartına özet/etiket + detay aç-kapa akışı eklendi;
  kullanıcıya atlanan sembollerin toplamı ve kod dağılımları net şekilde sunuldu. |
| 1.28 | 2026-08-19 | RSI uyarı paneli özet/etiket akışı kullanıcı deneyimine uygun olarak
  iyileştirildi ve panel mevcutsa detay açma doğrulaması UI smoke’da koşullu olarak
  eklendi. |
| 1.29 | 2026-08-19 | RSI uyarı özet etiketlerinin sıralaması sayısal önceliğe göre
  güncellendi; yetersiz RSI verisi (`insufficient_ohlcv_for_rsi`) kontrat senaryosu
  backend testlerine eklendi. |
| 1.30 | 2026-08-19 | RSI uyarı panelindeki detay görünürlük tercihi (göster/gizle)
  kalıcı hâle getirildi; kullanıcı tercihi localStorage ile sonraki oturumlarda
  korunur. |
| 1.31 | 2026-08-19 | `scripts/run_frontend_ui_smoke.mjs` RSI uyarı paneli detay
  görünürlüğünü kalıcılık yönünden doğrulayacak şekilde genişletildi; panel açıkken
  localStorage anahtarının doğru yazılması ve sayfa yenilemeyi takiben açık kalma davranışı
  test kapsamına alındı. |
