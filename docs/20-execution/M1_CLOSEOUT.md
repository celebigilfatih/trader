# M1 kapanış kontrolü

- **Tarih / durum:** 2026-11-06 — In Progress; bu kayıt M1'i kapatmaz.
- **İlk inceleme kaynağı:** `b6f07c6`; sonraki test/uygulama dilimleri aşağıda kayıtlıdır.
- **Amaç:** Mevcut kabul ölçütlerini kanıta bağlamak ve kalan P0 işlerini sıralamak.
  Yeni MVP kapsamı, mimari, sağlayıcı veya finansal politika tanımlanmaz.
- **Yetki:** [Roadmap](../00-product/ROADMAP.md), [milestone](MILESTONES.md),
  [gereksinimler](../00-product/REQUIREMENTS.md), BL-001/002/003 ve Accepted
  ADR-0004–0010. Constitution Draft olarak kalır.

## Güncel ilerleme — Accepted ADR-0010 uygulandı

[ADR-0010](../50-decisions/ADR-0010.md) A paketi açık kullanıcı onayıyla uygulandı:
VWEMA nedensel hacim ağırlığı, Supertrend sabit hazır olma, atomik input hatası
ve nötr eksik kart. VST-01–03 dar dilimi kapandı;19 core+3 API+4 kart testi eklendi.
Tam412 test /32.881 s /OK, dış socket engelli. Model/cache ve okuyucu sınırları
[uygulama kaydındadır](2026-11-06-adr0010-implementation.md). FVG/OB ve legacy
StockScreener, provenance ve diğer kapılar nedeniyle M1 In Progress kalır.

## Önceki VWEMA/Supertrend test dilimi

12 core+1 gerçek API testi eklendi; hedef27/27 ve tam **386 test /28.692 s /OK**,
dış socket engelli.32 geçerli transient mutant assertion failure/0 error;
restore27/27. Gerçek analyzer/ta ile yeterli veri ve sabit başlangıç alt kümesi,
public sinyal ve API okuyucuları sınandı; üretim/formül/config/ADR değişmedi.
[Plan/kanıt ve VST-01–03](2026-11-06-vwema-supertrend-tests.md).

VWEMA seed lookahead/başlangıç sonrası hacim tanımı ile Supertrend kısa veri
fiyat/yön/seed davranışı açık kalır; bunlar passing oracle yapılmadı. Sonraki
iş dar sözleşme/düzeltme kapsamını somutlaştırmak. FVG/OB aktif bağlantıları,
legacy StockScreener, provenance ve diğer entegrasyon kapıları nedeniyle M1 açık.

## Önceki OS-01 uygulaması

Somut A onay sorusuna verilen “Sıradaki Adım ile devam edelim.” yanıtıyla
ADR-0009 Accepted kaydedilip uygulandı. API `stoch` seçicisi mevcut `stoch_k`
serisine eşlenir; son %K skaler/null döner. Core/legacy formül ve K/D serileri,
diğer seçiciler, auth ve metadata korunur. OS-01'in bu dar dilimi kapandı.
7 yeni HTTP testi RED5 assertion failure/0 error →GREEN7/7; hedef38/38 ve
tam **373 test /27.902 s /OK**, dış socket engelli, asyncio loopback serbest.
10 geçerli transient mutant assertion failure/0 error; restore7/7.
[Uygulama ve kabul kanıtı](2026-11-06-adr0009-implementation.md).
Sonraki iş VWEMA/Supertrend ve aktif okuyucu bağlantıları; M1 In Progress kalır.

## Önceki sözleşme ve momentum incelemesi

OS-01 sonraki sözleşme incelemesi tamamlandı: [ADR-0009](../50-decisions/ADR-0009.md)
**Proposed**, `stoch` altında skaler %K önerir. Gerçek HTTP/fixture akışında
`stoch`500, Williams/CCI200, yeni olmayan K/D query adları400 ve auth401
doğrulandı; mevcut31 API/momentum testi geçti. Yeni eşleme uygulanmadı;
özel A onayı bekleniyor. [Tasarım ve kabul kaydı](2026-11-06-stochastic-api-design.md).

Stochastic/Williams %R/CCI çekirdek dilimi 17 gerçek hesaplama testiyle
doğrulandı; 34 geçerli bellek içi mutant assertion failure verdi, error yok.
Tam paket **366 test / 28.443 s / OK**, dış socket engelli; asyncio loopback
serbest. Çalışma zamanı/CI/manifest ve finansal politika değişmedi.
Plan ve ayrıntı [momentum execution kaydındadır](2026-11-06-momentum-indicator-tests.md).

**OS-01 açık:** Gerçek analiz route fonksiyonunda kontrollü provider ile
`indicators=stoch`, `KeyError('stoch')` üretir: hesaplayıcı `stoch_k/stoch_d`
üretirken API seçici anahtarı okumaktadır. Williams/CCI aynı reproducer'da
sayısal yanıt verir. Bu, çekirdek hesabın veya API Stochastic entegrasyonunun
tamamlandığı iddiası değildir. Hatalı API davranışı passing test yapılmadı.
Bu önceki incelemedeki OS-01 açıklığı sonraki Accepted uygulamayla yukarıda
kapatıldı; kayıt mevcut hatanın hâlâ sürdüğü şeklinde okunmamalıdır.

## Kabul ölçütü ve kanıt matrisi

“Dilim doğrulandı” yalnız belirtilen akış için geçerlidir; “kısmi” genel kapanış
kanıtının eksik olduğunu belirtir. Test sayısı bir kabul eşiği değildir.

| Mevcut M1 ölçütü | Kanıt | Durum / kalan iş |
|---|---|---|
| Ana hesaplamalar deterministik ve dış ağdan ayrılmıştır (REQ-002/003/007, BL-001) | 412 backend testi dış socket bağlantıları engelli geçti. Accepted ADR-0010 rasyonel/prefix/domain/readiness, gerçek sinyal/API ve kontrollü eksik kart kanıtı eklendi. | **Kısmi.** OS-01 ve VST-01–03 dar dilimleri kapandı; FVG/OB aktif bağlantıları ve legacy `StockScreener` zincirleri açık. P0-01/02. |
| Veri ve mock/deneysel kaynaklar açıkça etiketlenir (REQ-005/006, BL-002) | Cockpit analiz/RSI için 10 deterministik tarayıcı senaryosu; legacy haber/ML için 17 kontrollü AppTest. Accepted A metinleri, ADR-0004 zaman/skor ve ADR-0008 gözlem ufku uygulandı. | **Onaylı dilim doğrulandı; genel kapsam kısmi.** Diğer aktif legacy sonuçların kaynak/son bar/deneysel nitelik ve hata sonrası eski sonuç davranışı ekran envanterine bağlanmalı. P0-03. |
| Entegrasyon hatası görünürdür (REQ-001, BL-003) | OHLCV provider/adaptör ve RSS transport/tanı testleri; partial/failed RSS uyarıları ve cockpit veri yok/503/yeniden deneme senaryoları var. Sentetik fiyat veya başarısız NLP için sahte nötr skor üretilmez. | **Kısmi.** Legacy tarayıcı bazı istisnaları `continue` ile atlar; boş sonuç ile kesinti ayrımı ayrıca sınanmalı. Diğer aktif Yahoo yolları için bütçe/tanı ve okuyucu kapsamı netleşmeli. P0-02/04. |

Kaynak kanıtları: [test stratejisi](../30-quality/TEST_STRATEGY.md),
[BL-002 onaylı metinleri](../00-product/BL002_DATA_DISCLOSURE.md),
[ADR-0005 uygulaması](2026-11-05-adr0005-implementation.md),
[ADR-0007 uygulaması](2026-11-06-adr0007-implementation.md),
[ADR-0008 uygulaması](2026-11-06-adr0008-implementation.md).

## İlk kapanış kontrolünün tarihsel kanıtı — 349 test

- `.venv/Scripts/python.exe -B -` üzerinden unittest discovery: **349 test,
  53.874 saniye, OK**. Dış `connect`, `connect_ex` ve `create_connection`
  engellendi; Windows asyncio için loopback bağlantıları serbest bırakıldı.
  AppTest geçici dosyaları ve socketpair nedeniyle sandbox dışında çalıştırıldı.
- Aynı koşuda `sys.setprofile` ve `threading.setprofile` ile üç gerçek modüldeki
  metot çağrıları sayıldı. Bu bir satır/branch coverage ölçümü değildir; çağrılma
  sayısı sayısal doğruluğun veya tüm dalların sınandığının kanıtı değildir.
  Geçici JSON `.venv/m1-closeout-call-inventory.json` içindedir; sürümlenmez.
- `TechnicalAnalyzer`: Stochastic, Williams %R, CCI, `calculate_trend_strength`
  ve `generate_summary` **0 çağrı**; Supertrend **1**, VWEMA **2** çağrı.
  Son iki metot için mevcut çağrı/koruma veya sabit seri örneği, değişken pencere
  ve yön dönüşü için bağımsız aritmetik kanıtın yerine geçmez.
  Trend gücü/özet yardımcıları için modül dışında doğrudan Python çağıranı
  bulunmadı; bunlar önce aktif okuyucu açısından sınıflandırılır, kullanılmıyorsa
  gerekçeli N/A bırakılır. Sıfır çağrı her yardımcıya otomatik P0 testi yüklemez.
- `_calculate_fvg` ve `_calculate_order_block` sarmalayıcıları **0 çağrı**;
  alt patern sınıflarının testleri vardır. Bu, bütün FVG/OB hesabı testsiz demek
  değildir; seçici → sarmalayıcı → okuyucu bağlantısı ayrıca doğrulanmalıdır.
- Gerçek `StockScreener` kurucusu ve tüm metotları **0 çağrı**. `app.py` RSI,
  teknik/boğa taramaları ve haftalık/aylık performans için bu sınıfı kullanır.
  API RSI testleri farklı yolu sınar; legacy tarayıcı kapsamı sayılmaz.
- Gerçek `BISTDataFetcher.get_stock_data` **6 çağrı**; `get_real_time_data`,
  `get_multiple_stocks`, `get_bist_index_data`, `validate_symbol`,
  `get_company_info` **0 çağrı**. Önce aktif okuyucular saptanmalıdır; kullanılmayan
  yardımcıların varlığı tek başına yeni bir ürün veya timeout gereksinimi yaratmaz.
- Next.js build ve 10 tarayıcı senaryosu bu belge adımında yeniden çalıştırılmadı.
  Son checkpoint `abfc592` öncesindeki başarılı üretim build/10 Edge senaryosu
  tarihsel kanıttır; frontend kaynakları o zamandan beri değişmedi. AppTest tam
  monolit veya canlı sağlayıcı testi değildir; finansal doğruluk ölçülmedi.
- 62 sürümlü Python dosyası bytecode yazmadan `compile()` ile kontrol edildi;
  39 zorunlu CDSK dosyası, 11 Boot bölümü, 8 Accepted ADR ve değişen belgelerin
  yerel bağlantıları doğrulandı. Frontend'in `abfc592` ile aynı olduğu ve bu
  adımda runtime/test/CI kaynaklarının değişmediği Git diff ile kontrol edildi.
- Bağımsız salt okunur incelemede Critical/Important bulgu yok. Kullanılmayan
  trend/özet yardımcılarına otomatik P0 kapsamı yüklememek için aktif okuyucu/N/A
  sınırı açıklaştırıldı; finansal etkinlik ve uzak CI doğrulaması yapılmadı.

## Kalan P0 işleri — uygulama sırası

Bu alt işler mevcut BL-001/002/003'ü ayrıştırır; bağımsız yeni ürün taahhüdü
değildir. Her dilimde önce literal/elle hesaplanan oracle ve sınır fixture'ı,
sonra gerekirse dar düzeltme, ilgili regresyonlar ve belge/handoff güncellemesi
beklenir. Bilinen bir kusur passing beklenti olarak sabitlenmemelidir.

| Sıra / kayıt | Kapsam | Somut kapanış kanıtı | Karar sınırı |
|---|---|---|---|
| **1 — M1-P0-01 / BL-001** | Momentum/OS-01 ve VST-01–03 dar dilimleri tamamlandı. FVG/OB aktif okuyucu bağlantıları kalır; trend gücü/özet aktif kullanım veya gerekçeli N/A açısından sınıflandırılır. | Seçili gerçek hesaplayıcıda değişken OHLCV, pencere/warmup ve uygun eşik/yön dönüşü örnekleri; girdiyi koruma ve gerçek okuyucu sözleşmesi. Her alt dilim ayrı kaydedilir; yalnız çağrı sayısını artırmak yeterli değildir. | Mevcut davranışın doğrulaması yeni ADR gerektirmez. Finansal formül, warmup veya geçersiz girdi politikası değişecekse somut karar ve AGENTS onay kapısı uygulanır. |
| **2 — M1-P0-02 / BL-001/003** | UI'nın gerçekten çağırdığı legacy `StockScreener` yolları; önce RSI, ardından boğa taramaları ve haftalık/aylık performans. | Gerçek analyzer + enjekte OHLCV ile dahil/dışarıda/eşitlik/sıralama; interval-period iletimi; boş/yetersiz veri, bir sembolün hatası ve tüm kaynakların kesintisi. Sağlayıcı ağı ve gerçek sleep kullanılmaz; boş sonuç/kesinti görünürlüğü ayrıca kontrol edilir. | Yeni çıktı alanı veya hata sözleşmesi gerekiyorsa önce somutlaştırılır; mevcut kusur sessizce doğru kabul edilmez. |
| **3 — M1-P0-03 / BL-002** | Aktif legacy analiz/tarama sonuçlarını onaylı kaynak/deneysel metinler ve hata sonrası durumla eşleştiren ekran envanteri. | Her aktif sonuç için kaynak, veri zamanı, veri niteliği ve eksik/hatalı durumda gösterim kanıtı veya gerekçeli N/A; yanlış “canlı/gerçek” iddiası ve eski sonucun güncel sunulması kalmaz. Bağlanmayan mock yardımcılar ayrı kaydedilir. | Kabul edilmiş metinlerin uygulanması kapsamındadır; yeni ekran/model/metadata sözleşmesi ayrıca onay gerektirebilir. |
| **4 — M1-P0-04 / BL-003** | OHLCV provider dışındaki aktif Yahoo/SDK çağrılarını ve hata okuyucularını sınıflandırma. | Her aktif yol için mevcut timeout/retry/hata dönüşü ve kullanıcı görünürlüğü; kontrollü başarı/boş/kesinti testi. Kullanılmayan yöntemler gerekçeli N/A. `.info` kullanan yardımcılar ortak OHLCV bütçesi altında diye raporlanmaz. | Yeni SDK bütçe/retry/transport politikası Accepted ADR-0005'in otomatik uzantısı değildir; gerekirse Proposed ADR ve açık onay. Üretim merkezi monitoring bu dilime eklenmez. |

**Hemen sonraki adım:** Kalan FVG/OB aktif okuyucu ve legacy tarayıcı bağlantılarının
mevcut sözleşmesini gerçek hesaplayıcı testleriyle doğrulamak.
Çekirdek testlerinin geçmesi bütün okuyucuların çalıştığını göstermez.

## Tekrarlanabilirlik ve CI sınırı

- Sürümlü temel komut `make test-backend` veya
  `.\.venv\Scripts\python.exe -B -m unittest discover -s backend/tests -p 'test_*.py'`.
  Bugünkü tam koşuda kullanılan dış socket guard'ı geçici çalıştırıcıdadır;
  Makefile genel ağ engelleme kapısı uygulamaz. Bu nedenle komut tek başına
  gelecekteki kazara ağ erişimini engelliyor diye sunulamaz. Kalıcı çevrimdışı
  çalıştırıcı/CI bağlantısı, REQ-007'nin tekrarlanabilir kanıtını güçlendirecek
  kalite işidir; yalnız bugünkü test toplamına dayanarak kapatılamaz.
- `.github/workflows/ci.yml`: Ubuntu, Python 3.11, Node 18, **yalnız `make ci`**;
  PR ve `main` push tetiklenir. Bu feature dalına push tek başına CI koşusu
  kanıtı değildir; bu adımda uzak CI çalışması doğrulanmadı.
- `make ci` bağımlılık kurulumu, backend testleri, frontend build ve stack/API
  smoke çalıştırır. `ci-e2e`/`ci-full` yerel alternatiflerdir; mevcut workflow
  bunları ve 10 deterministik browser senaryosunu çalıştırmaz. Stack smoke
  sağlayıcı dayanıklılığı veya deterministik tarayıcı kapsamının yerine geçmez.
- Deterministik browser hedefini otomasyona bağlamak mevcut testlerin korunması
  için yararlıdır; M1'e yeni browser altyapısı şartı eklenmez. Canlı smoke ve
  çevrimdışı kabul kanıtı ayrı raporlanmalıdır.
- Yerel doğrulama Python 3.12.14 ile yapıldı; bu, CI Python 3.11 doğrulaması
  değildir. `npm install` ve Python manifestleri için destek/pin politikası
  BL-007'dir; bu belge dependency veya destek matrisi kararı almaz.
- CDSK upstream validator'ın belgelenmiş macOS yolu ve komşu
  `D:/Dev/cdsk/scripts/validate_project.py` bu ortamda yoktur. 39 zorunlu dosya
  ve 11 Boot bölümü yapısal kontrolü upstream validator başarısı sayılmaz.
  Bu kalite kapısı için çalışabilir upstream validator kanıtı hâlâ eksiktir.

## M1 dışındaki açık işler ve kapanış kararı

- M2: gerçek davet/kimlik operasyonu, paper ürün doğrulaması, alarm/bildirim ve
  günlük snapshot. M3: PostgreSQL/worker/Coolify, backup/restore, rollback,
  merkezi monitoring ve operasyon sahipliği. Bunların tamamı M1'e taşınmaz.
- BL-004/005/007: README/lisans doğruluğu, artifact hijyeni, sürüm/dependency
  politikası. Veri kullanım koşulları ve üretim lisansı gerçek dağıtım öncesi
  ayrıca doğrulanmalıdır; Accepted mimari kararı lisans doğrulaması değildir.
- `detect_chart_patterns` içindeki head/shoulders, triangle ve flag alanları
  yalnız başlangıç `False` değerini korur. Bunlar için olumlu tespit yeteneği
  kanıtı yoktur; BL-004 ürün iddiası envanterinde ayrıştırılır. Sabit `False`
  davranışını test etmek bu yeteneklerin tamamlandığını göstermez; yeni algoritma
  kapsamı bu kapanış kontrolünde onaylanmış sayılmaz.
- ML finansal accuracy/kalibrasyon, UI yayılım heuristiği, Next.js kurulum
  güvenlik uyarısı ve Starlette/Streamlit deprecation kayıtları sürer.

M1 ancak yukarıdaki üç mevcut kabul ölçütü için kalan aktif yollar kanıtlandığında
veya yetkili kapsam kararıyla gerekçeli N/A olduğunda yeniden değerlendirilebilir.
P0 test/hata görünürlüğü veya veri niteliği açığı varken **Completed** yapılmaz.
Genel Definition of Done şablonu bu belgeyle topluca işaretlenmez. İlk kapanış
kontrolü yalnız belgeleri değiştirdi; sonraki momentum dilimi 17 test ekledi.
Sonraki Accepted ADR-0009 yalnız API alias'ı ve7 HTTP testi ekledi; CI,
kalıcı veri modeli, bağımlılık ve diğer Accepted ADR'ler değişmedi.
VWEMA/Supertrend adımı13 test daha ekledi; üretim politikası değiştirilmedi.
