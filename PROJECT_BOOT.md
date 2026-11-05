# BIST Teknik Analiz Uygulaması — Project Boot

- **Document version:** 0.1
- **CDSK version:** 0.1.0
- **Last updated:** 2026-11-05

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

- **Phase:** Kullanıcı isteğiyle geliştirmeye ara verildi (2026-11-05);
  yeniden devam talebi bekleniyor. Uygulama süreçleri bu belge işlemiyle durdurulmadı.
- **Version:** Release etiketi yok (`TBD`).
- **Active sprint / milestone:** M0 CDSK başlangıcı tamamlandı; M1 güvenilirlik
  tabanı başladı. Sonraki adım API-first Next.js cockpit’i ile uçtan uca akışın
  stabilizasyonu.
- **Current focus:** FastAPI çekirdeği ile Next.js MVP cockpit’i arasındaki
  kontratların senkronizasyonu; auth, teknik analiz, watchlist ve portföy akışlarının
  doğrulanması.
- **Critical risks:** Finansal çıktının yanlış yorumlanması; mock/gerçek veri
  karışması; dış servis kesintisi; ağ bağımlı ve yetersiz testler; `.next` gibi
  frontend build artefaktı yönetimi; açık veri lisansı/dağıtım modeli olmaması.
- **Blocking decisions:** Test kapsamının kapsamlı kapanış planı; gerçek kullanıcı
  erişim ve davet operasyona geçildiğinde operasyon sahipliği tanımı.
- **Last status update:** 2026-11-05

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
- **Recent decisions:** ADR-0001, ADR-0002, ADR-0003 kabul edildi; target mimari ve
  kapsam netleştirildi.
- **Recently completed:** ADR zinciri, migration hedef resmi ve roadmap/milestone
  güncellemesi tamamlandı.

## 7. Current Priorities

1. Market-data ve API kontratlarında ek offline deterministik senaryoları tamamlamak.
2. Next.js cockpit’de auth, hata yönetimi, boş veri ve retry akışlarını sağlamlaştırmak.
3. Paper portföy ve karar cockpit’i için write operasyonlarını (ekleme/çıkarma) ve
   senaryo testlerini eklemek.
4. Yahoo/RSS entegrasyonlarına timeout/retry ve görünür hata ölçümü eklemek.
5. Davetli beta kimlik ve in-app/e-posta alarm hattını açmak.

## 8. Known Constraints

- 2026-08-19 itibarıyla, `make ci` akışı backend çalışabilirlik ve smoke
  kontrolünü geçebilecek şekilde stabilize oldu; çevrimdışı deterministik test ve
  veri güvenilirliği açmazları halen izleniyor.
- Frontend build ve bağımlılık adımlarında `npm` kritik güvenlik zafiyetleri bulunuyor;
  CI’nin bu adımı başarısız olmadan önce risk notu tutuldu.
- Repository geçmişten kalan `__pycache__/*.pyc` dosyalarını izliyor.
- Constitution `Draft`; bağlayıcı hale gelmesi açık ürün sahibi onayı gerektirir.

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
- [x] CDSK validator ve Python sözdizimi geçti; `make ci` hedefi backend testleri,
  frontend build ve run-dev smoke ile başarılı şekilde tamamlandı.
- [x] İlgili dokümantasyon güncellendi.
- [x] ADR ihtiyacı ve changelog tamamlandı.
- [x] Repository tutarlılığı doğrulandı.
- [x] Açık riskler ve sonraki adım kaydedildi.
- [x] Çevrimdışı, deterministik market-data ve ana API akış testleri eklendi
  (`backend/tests/test_market_data.py`, `backend/tests/test_api_contracts.py`) ve
  2026-11-05 itibarıyla toplam backend test sayısı 35'e çıktı. Bu, M1'in
  bütün kabul kriterlerinin kapandığı anlamına gelmez.

## 11. AI Handoff

### Güncel devir — 2026-11-05

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
