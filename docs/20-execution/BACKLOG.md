# Backlog

## Devam ve kaldığımız yer — 2026-11-06

FVG/OB sarmalayıcı→legacy chart bağlantısı11 testle doğrulandı ve DataFrame
okuyucu kusuru düzeltildi; gerçek legacy RSI ilk dilimi6 core+2 actual buton
AppTest ile sınandı. [Kanıt/kapsam](2026-11-06-fvg-ob-legacy-rsi.md), hedef45/45.
OS-02 çoklu çıktı API500 ve RS-01 kesinti/no-match ayrımı açık:
[ADR-0011](../50-decisions/ADR-0011.md) ve
[ADR-0012](../50-decisions/ADR-0012.md) **Proposed**, özel A kararları bekleniyor.
Boğa/teknik ve haftalık/aylık legacy tarayıcılar, provenance ve genel M1 açık.

### Önceki ADR-0010 uygulaması

[ADR-0010](../50-decisions/ADR-0010.md) A paketi açık onayla Accepted ve uygulandı.
VST-01–03 dar dilimi kapandı: nedensel hacim ağırlıklı VWEMA, Supertrend n<=p'de
iki eksik alan/default ilk hazır indeks10, strict girdi/atomik ValueError ve
eksik VWEMA kartında nötr —/Hesaplanamadı. Yeni19 core+3 API+4 kontrollü kart testi;
tam412/412 dış socket engelli (32.881 s), önceki geçerli13 test korunur.
[Uygulama/kanıt](2026-11-06-adr0010-implementation.md). Optional eski model
yeniden eğitim/cache yenileme sınırları ADR'dedir; kullanıcı artifact'ı değişmedi.
Sıradaki iş FVG/OB aktif okuyucu bağlantıları ve legacy StockScreener zincirleri.
Genel M1 In Progress; canonical/finansal etkinlik doğrulanmış değildir.

### Önceki VWEMA/Supertrend test dilimi

VWEMA/Supertrend geçerli hesaplama ve okuyucu dilimine13 test eklendi:
12 gerçek core,1 gerçek API. Hedef27/27 ve tam386 backend testi dış socket
engelli /28.692 s /OK;32 geçerli transient mutant assertion failure/0 error.
Core/ta/pandas, public generate_signal ve VWEMA API okuyucuları gerçek;
yalnız ATR exception fallback'ine kontrollü hata enjekte edildi. Üretim değişmedi.
[Plan/kanıt ve açık VST-01–03](2026-11-06-vwema-supertrend-tests.md): VWEMA seed
lookahead/pozitif hacmin sadeleşmesi; Supertrend kısa veri fiyat/yön ve
seed geçmişi tutarsızlığı ayrı karşı örneklerle kayıtlı. Bunlar passing
beklenti yapılmadı; sonraki iş dar sözleşme/düzeltme kapsamını somutlaştırmak.
FVG/OB aktif bağlantıları/legacy tarayıcılar ve genel M1 açık kalır.

### Önceki ADR-0009 uygulaması

Somut ADR-0009 A onay sorusuna “Sıradaki Adım ile devam edelim.” yanıtı bu
eşlemeye uygulama onayı kaydedildi; **Accepted ve uygulandı**. Yalnız API
projeksiyonunda `stoch` → `stoch_k`; dış skaler/null ad, core/legacy, auth ve
diğer seçiciler aynı. OS-01 dar dilimi kapandı; genel M1 açık.
7 yeni gerçek HTTP testi önce5 assertion failure/0 error, sonra7/7; mevcut
API/momentum ile38/38 ve tam373 backend testi dış socket engelli /27.902 s /OK.
10 geçerli transient mutant assertion failure/0 error; restore7/7.
Plan/kanıt [execution kaydında](2026-11-06-adr0009-implementation.md).
Sıradaki iş M1-P0-01'in kalan VWEMA/Supertrend hesapları ve aktif okuyucu bağlantıları.

### Önceki OS-01 sözleşme tasarımı

OS-01 için [ADR-0009](../50-decisions/ADR-0009.md) **Proposed** hazır:
`data.indicators.stoch` = son %K (float/null); K/D alanları veya yeni query
seçicisi eklenmez. K≠D/zero/null/mixed/meta/400/401 kabul ölçütleri
[tasarım kaydındadır](2026-11-06-stochastic-api-design.md). Özel A onayı bekleniyor;
genel devam talebi bu önceden tanımlanmamış semantik seçimin onayı sayılmadı.
Gerçek HTTP `stoch`500 ve komşu yollar yeniden gözlendi; mevcut31 API/momentum
testi dış socket engelli geçti. Bu belge adımında runtime/test/CI/config değişmedi.
Sonraki adım özel karar, ardından dar TDD düzeltmesi; OS-01 ve M1 açık.

### Önceki momentum test dilimi

M1-P0-01 Stochastic/Williams %R/CCI çekirdek dilimi tamamlandı: 17 yeni gerçek
hesaplama testi; 34 geçerli transient mutant assertion failure/0 error. Tam
366 backend testi dış socket engelli / 28.443 s / OK. Girdi/index/prefix,
pencere, config, warmup/flat ve gerçek sıfır ayrımı sınandı; runtime değişmedi.
Plan/kanıt [momentum kaydındadır](2026-11-06-momentum-indicator-tests.md).

**OS-01 — açık P0 okuyucu kusuru:** `analyze_symbol(..., indicators='stoch')`
gerçek hesaplayıcı + kontrollü provider ile `KeyError('stoch')` verir.
Üretici `stoch_k/stoch_d`, API okuyucusu seçici `stoch` anahtarını kullanır.
Bu kusur passing beklenti yapılmadı; o aşamada gereken sözleşme incelemesi
ve somut öneri yukarıda tamamlandı, özel onay bekliyor.
Ardından VWEMA/Supertrend ve diğer aktif okuyucu boşlukları kalır. M1 açık.

### Önceki M1 kapanış kontrolü

M1 kabul kapısı kontrolü tamamlandı; milestone **In Progress** kalır.
[Kapanış matrisi ve sıralı P0 işleri](M1_CLOSEOUT.md): önce Stochastic/Williams
%R/CCI, sonra diğer hesap/okuyucu boşlukları, gerçek legacy tarayıcı zincirleri,
aktif ekran provenance envanteri ve diğer aktif Yahoo hata yolları.
349 test dış socket engelli yeniden geçti; çağrı envanterinde ilk üç indikatör
ve bütün `StockScreener` metotları sıfır çağrı verdi. Bu bir satır/branch coverage
veya finansal etkinlik ölçümü değildir. REQ-004 Accepted ADR-0004 ile uyumlandı;
REQ-006/007 kısmen doğrulandı. CI yalnız `make ci` çalıştırır; README düzeltildi.
Üretim kodu/test/CI/dependency/ADR o kontrol adımında değişmedi. O aşamadaki
M1-P0-01 ilk test dilimi yukarıda tamamlandı.

### Önceki ADR-0008 uygulama dilimi

Kullanıcı somut ADR-0008 sorusuna “onayladım, devam edelim” dedi; Accepted ve
uygulandı. Tek ufuk cap'i, price/current_price helper yolu+legacy uyumu ve panelde
h gözlem/gerçek son index/0..h şematik eksen tamamlandı.18 yeni testle349 backend,
97 ML/17 kontrollü Streamlit testi;19 geçerli mutant ve bağımsız inceleme kanıtı
[execution kaydındadır](2026-11-06-adr0008-implementation.md).
Fiyat/format1/HTTP/model politikalarının onay dışı kısımları değişmedi; M1 genel
kapanışı iddia edilmez. O aşamadaki sonraki iş olan M1 kabul kapıları ve kalan
P0 kapanış kontrolü yukarıda tamamlandı; UI yayılım kalibrasyonu ayrı kalır.

Kullanıcı Commit & Push istedi;70 dosyalık tamamlanmış çalışma `abfc592` ile
`origin/codex/m1-pattern-signal-tests` dalına gönderildi. Commit öncesi331 backend
testi, Next.js üretim build ve10/10 deterministik Edge senaryosu tekrar geçti.
Sonraki ML-07/08/gözlem ufku için [ADR-0008](../50-decisions/ADR-0008.md)
o aşamada **Proposed** hazırlandı; bağımsız builder incelemesi önemli açık soru bulmadı.
Tek interval cap, fiyat/current_price helper yolu+legacy uyumu ve kesin tarih
yerine gözlem sunumu özel açık onay bekliyordu. Sonraki uygulama yukarıdadır.
Kayıt [tasarım/teslim belgesindedir](2026-11-06-ml-horizon-confidence-design.md).

Simple ML-04–06 dar düzeltmeleri tamamlandı: her eğitim eski trained/cache
durumunu iptal eder, başarısız metric başarı sayılmaz; importance gerçek RF
feature_names_in_ metadata'sını kullanır. Nonfinite/pozitif olmayan fiyat ve
taşan sonuçlar error döner; eski tahmin cache'i hata sonrasında kalmaz.
13 yeni test önce29 assertion failure/0 error, sonra37 hedef test green;
11 geçerli transient mutant yakalandı. Bağımsız inceleme bulgusuz.
Plan/kanıt [Simple ML kaydındadır](2026-11-06-simple-ml-fixes.md).
Yeni ADR, çıktı/kalıcı şema, model/config/dependency değişikliği yoktur.
Sonraki adım ML-07 ufuk cap ve ML-08 confidence sözleşmesini somutlaştırmak;
finansal politika/API değişirse AGENTS onay kapısı uygulanır. Genel M1/M2 açık.

Kullanıcı somut ADR-0007 sorusuna “Onaylıyorum.” dedi; Accepted ve uygulandı.
ML-01–03 ile advanced ML-04 onaylı kapsamı kapandı: NaN hedef/purge, train-only
bounds, ortak dört-model scaler/inverse, coherent state ve format1 joblib.
30 yeni regresyonla70 hedef ML testi; iki bağımsız review bulgusu RED→GREEN.
Eski sürümsüz model yeniden eğitim ister; kullanıcı dosyası migrate edilmedi.
Kapsam/kanıt [execution kaydındadır](2026-11-06-adr0007-implementation.md).
O dilimin sonraki Simple ML-04–06 adımı yukarıda tamamlandı; ML-07/08 politika/API
değişikliği ayrıca değerlendirilir. Genel M1/M2/accuracy kapanmadı.

Önceki tasarım aşamasında [ADR-0007](../50-decisions/ADR-0007.md)
**Proposed** hazırlandı. Öneri: bilinmeyen hedefler NaN; ufka göre train/test purge;
train-only %1/%99 bounds ve her modelde ortak scaler/inverse; tutarlı advanced
retrain state ve versioned artifact. Eski sürümsüz artifact yeniden eğitim
gerektirecekti. O aşamada AGENTS açık onayı bekleniyordu;
genel devam talebi Accepted sayılmadı. Sonraki açık onay/uygulama yukarıdadır.
Tasarım, kabul ölçütleri ve kanıt:
[2026-11-06-ml-correction-design.md](2026-11-06-ml-correction-design.md).
O aşamanın sonraki adımı onay ve RED→GREEN uygulamaydı; güncel durum yukarıdadır.

BL-001 ML yardımcıları dar test dilimi tamamlandı: 9 ortak + 15 simple + 16
feature/eğitim = 40 yeni test, toplam 288 backend testi. 80/80 geçerli transient
mutant assertion failure verdi; bağımsız inceleme 40 testi yeniden doğruladı.
Üretim ML kodu değişmedi. Sonraki öncelik aşağıdaki ML-01–03 hesaplama kusurları
için dar düzeltme planı; model/feature/veri sözleşmesi değişecekse somut ADR ve
AGENTS onay kapısı uygulanır. Genel BL-001/M1 kapanmadı.
Plan/kanıt: `2026-11-06-ml-helper-tests.md`.

Kullanıcı ADR-0006 için “onaylıyorum” dedi; Accepted kaydedilip uygulandı.
OTT başlangıcı loop öncesine taşındı, `ott_trend` ilk gözlemde eksik ve sonra
±1 olarak üretiliyor. Gerçek sinyal/pozisyon okuyucuları iki yönü tüketiyor.
RP-03 onaylı kapsamı kapandı; ardından ML test dilimi yukarıda tamamlandı.
Plan/kanıt: `2026-11-06-adr0006-implementation.md`.

RP-01/02/04/05 bilinen hata örnekleri düzeltildi: sıfır göstergeler mevcut
kurallara girer, güçlü SAT mevcut satış boyutunu kullanır; DayTrader geçersiz
entry/ATR/yön girdisini ValueError ile sonuç/kayıt üretmeden reddeder. OTT
pandas runtime hatası `.ffill()` ile çözüldü; seed ve `ott_trend` sözleşmesi
o aşamada uygulanmamıştı; sonraki onaylı uygulama yukarıda kayıtlıdır.
20 regresyon/232 test o önceki dilimin kanıtıdır:
`2026-11-06-risk-position-fixes.md`.

BL-001 risk/pozisyon test dilimi: 13 risk + 16 pozisyon + 17 DayTrader testi
eklendi. Bağımsız sayılarla risk/stop/hedef, ATR/entry/R:R ve yerel P&L sınandı;
104/104 geçerli bellek içi mutant yakalandı. Plan ve reproducer'lar:
`2026-11-06-risk-position-tests.md`. Açık sıfır gösterge/strong-SAT/OTT bulguları
nedeniyle risk/pozisyon entegrasyonunun bütünü o aşamada kapanmadı. Sonraki
düzeltme ve kalan OTT onay kapısı yukarıda kayıtlıdır; ardından ML yardımcıları.

BL-001 temel BOS/kombinasyon dilimi tamamlandı: 15 BOS + 12 kombinasyon testi,
166 birleşik test ve 40/40 yakalanan geçerli bellek içi mutant. Gerçek OHLCV
zincirleri ve ayrı matcher sınırları doğrulandı; finansal hesaplar değişmedi.
Plan/kanıt: `2026-11-06-bos-combo-tests.md`. Sonraki risk/pozisyon dilimi yukarıda
kayıtlı; genel BL-001/M1 kapanışı ve ek yapı kenarları açık.

BL-001 MACD/Bollinger değişken seri açığı kapatıldı: 5 yeni test, 17 hedef/
139 birleşik test ve 12/12 yakalanan bellek içi mutant. Mevcut finansal hesaplar
değişmedi. Plan/kanıt: `2026-11-06-macd-bollinger-tests.md`.
Bunun ardından BOS/kombolar yukarıdaki dar dilimde doğrulandı.

Kullanıcının projeyi inceleyip devam etme talebiyle önceki ara kaldırıldı.
AGENTS.md okuma sırası ve Git durumu doğrulandı; sıralı iş 2 ve 3 için dar test
adımları tamamlandı. Önceki patern/sinyal değişiklikleri korundu. Yeni commit/push
yapılmadı. Plan/kayıtlar: `2026-11-05-pattern-signal-tests.md` ve
`2026-11-05-deterministic-ui-tests.md`.
BL-002 için kaynak incelemesi ve kesin metin önerisi
`docs/00-product/BL002_DATA_DISCLOSURE.md` içinde hazırlandı. RSS zaman/NLP hata
ayrımı `ADR-0004` olarak hazırlandı. Kullanıcının doğrudan onay sorusuna
“devam et” yanıtıyla A metinleri ve ADR-0004 kabul edildi ve uygulandı.
Plan/kanıt: `2026-11-05-bl002-implementation.md`.
BL-003 mevcut hata-yolu diliminde Yahoo timeout iletimi ve RSS bilinen HTTP hata
gövdesi düzeltildi; önceki dilimde 21 test eklendi (toplam 103). Kullanıcı somut
onay sorusuna “devam edelim” yanıtıyla `ADR-0005` politikasını kabul etti.
RSS bütçe/tanı ve legacy kaynak görünürlüğü uygulama kaydı:
`2026-11-05-adr0005-implementation.md`.

- **CDSK:** 0.1.0 belge/çalışma iskeleti kurulu; M0 tamamlandı olarak kayıtlı.
  Yapısal validator'ın geçmesi, tüm ürün veya operasyon gereksinimlerinin
  tamamlandığı anlamına gelmez. Constitution hâlâ Draft ve TBD alanları var.
- **M1:** Kısmi; 431 backend testi (FVG/OB/legacy RSI/VWEMA/Supertrend/Stochastic API/momentum/ML/OTT/RSS/NLP/provider ve kontrollü Streamlit dahil)
  dış ağ engelli geçti. Cockpit seti 10/10 deterministik senaryodur. İlk dört
  işin ve onaylı RSS politikasının dar dilimleri uygulandı; BL-001 ve genel kapanış kriterleri açık.
- **M2:** Yerel demo auth, watchlist ve paper portföy temeli var; gerçek davet
  operasyonu, alarm/bildirim ve günlük snapshot kapanışı açık.
- **M3:** Üretim hazırlığı Proposed; PostgreSQL kalıcılığı, worker, deployment,
  backup/restore, rollback ve operasyon sahipliği tamamlanmadı.
- **Git:** `codex/m1-pattern-signal-tests`; bu kontrol başlangıcında HEAD
  `b6f07c6`, çalışma ağacı temiz ve dal upstream ile eşitti. Önceki açık Commit &
  Push yetkisiyle `abfc592`, `fbc7c43` ve teslim belgeleri aynı dala gönderildi.
  Daha eski “commit/push yok” notları kendi adımlarının tarihsel kaydıdır.

| ID | Outcome | Priority | Dependencies | Acceptance criteria | Status |
|---|---|---|---|---|---|
| BL-001 | Deterministik çekirdek testleri | P0 | Test fixture'ları | Teknik analiz, patern, sinyal ve ML yardımcıları ağ olmadan test edilir. | In Progress |
| BL-002 | Gerçek/mock veri ayrımı | P0 | A metinleri ve ADR-0004 onaylandı | Mock sentiment ve deneysel ML her ilgili ekranda görünür etiketlenir. | Onaylı dilim uygulandı; genel legacy/kapanış kapsamı In Progress |
| BL-003 | Dış entegrasyon dayanıklılığı | P0 | ADR-0005 Accepted | Yahoo/RSS çağrılarında timeout, sınırlı retry/backoff ve hata metriği bulunur. | OHLCV/RSS yerel dilimi uygulandı; diğer Yahoo yolları ve üretim gözlemi açık |
| BL-004 | README ve lisans doğruluğu | P1 | Ürün sahibi | Placeholder URL/iletişim temizlenir; lisans iddiası gerçek LICENSE ile uyumlu olur. | Backlog |
| BL-005 | Repository hijyeni | P1 | Veri sahipliği kararı | İzlenen `__pycache__` dosyaları ve orphan veri/artifact'lar güvenli biçimde ele alınır. | Backlog |
| BL-006 | `app.py` modülerleştirme planı | P1 | Mimari ADR gerekebilir | UI sayfaları ve orchestration küçük, test edilebilir modüllere ayrılacak planla tanımlanır. | Proposed |
| BL-007 | Bağımlılık ve Python sürümü tabanı | P1 | Destek matrisi kararı | Desteklenen Python sürümü ve yeniden üretilebilir dependency politikası doğrulanır. | Proposed |
| BL-008 | Veri/lisans ve dağıtım kararı | P0 before production | Ürün sahibi + hukuk/operasyon | Sağlayıcı koşulları, hosting, privacy ve rollback Accepted ADR'larla kayıtlıdır. | Accepted |
| BL-009 | Deploy hedefi ve operasyon hattı kurulumu | P0 | ADR-0002 | Coolify cron, PG backup, health ve rollback ile günlük snapshot iş hattı kurulur. | In Progress |

### Risk/pozisyon bulgularının güncel durumu

İlk reproducer'lar risk/pozisyon test kaydındadır; düzeltme kanıtı
`2026-11-06-risk-position-fixes.md` içindedir. Puan/formül/bağımlılık politikası
değişmedi; OTT alan/seed önerisinin sonraki Accepted uygulaması RP-03'tedir.

| Kayıt | Gözlenen davranış | Sonraki işlem / sınır |
|---|---|---|
| RP-01 | Çözüldü: RSI/MACD sıfırları mevcut kurallara girer; None/NaN hâlâ atlanır. | 8 yeni alert regresyonu ve gerçek düşen RSI zinciri; puan ağırlığı değişmedi. Sonsuz gösterge politikası ayrı. |
| RP-02 | Çözüldü: GÜÇLÜ SAT mevcut “Pozisyonun %75-100'ünü sat” dalına girer. | Güçlü AL ve diğer eski dallar korunur. UI tüketimi iddiası yok; mevcut satış yüzdesi kullanıldı. |
| RP-03 | Çözüldü: Accepted ADR-0006 seed/yön ve boş/tek/geçersiz Close sözleşmesi uygulandı; pandas runtime uyumlu. | Gerçek üretici→crossover/pozisyon okuyucusu iki yönde doğrulandı. Bu prototip recurrence'idir; canonical OTT/etkinlik araştırması veya üretim kapanışı değil. |
| RP-04 | Çözüldü: yetersiz veri veya pozitif/sonlu olmayan entry fiyatı/ATR, indicator işinden önce ValueError verir. | ATR helper sıfır aralığı hesap olarak korur; entry için pozitif ATR gerekir. Gerçek scan hatalı sembolü atlayıp devam eder. |
| RP-05 | Bilinen yollar çözüldü: boş/kısa ATR ValueError; add/close yalnız BUY/SELL, hatalı kayıt/yön mutasyondan önce reddedilir. | Miktar/sermaye/diğer bozuk alanlar veya desteklenmeyen ATR parametreleri için yeni kural seçilmedi. |

### ML bulgularının güncel durumu — 2026-11-06

Sentetik runtime reproducer'ları test kaydındadır. Hatalı sonuçlar passing golden
assertion olarak benimsenmedi. Sonraki Accepted ADR-0007 uygulaması ML-01–03 ve
advanced ML-04 onaylı kapsamını düzeltti. Sıralı split'te hedefin test aralığına
uzanması da kaynak-konum purge regresyonuyla kapandı; kalan kayıtlar açıktır.

| Kayıt | Gözlenen davranış | Sonraki işlem / sınır |
|---|---|---|
| ML-01 | Çözüldü: bilinmeyen/sonlu olmayan veya pozitif olmayan fiyat hedefi NaN; gerçek sıfır getiri korunur. | Accepted ADR-0007 native hedef/50 etiket ve ufuk purge regresyonları; terminal satırlar eğitimde etiketli sayılmaz. |
| ML-02 | Çözüldü: tüm estimator inference girdisine stored bounds/X scaler ve çıktısına target inverse uygulanır. | Dört gerçek modelle literal160→164 label-boundary testi; native dört-model smoke ayrı, accuracy kanıtı değil. |
| ML-03 | Çözüldü: stateless feature hazırlığı; quantile bounds yalnız purge edilmiş train üzerinde bir kez fit edilir. | Heldout değişimi fitted bounds/scaler değiştirmez, tahminde yeniden fit yok. Caller indicator causal garantisi veya tam backtest kapanışı değil. |
| ML-04 | Çözüldü: advanced ve simple yeni train denemesi eski trained/cache durumunu iptal eder. Simple sonlu olmayan metrikte de başarı yayımlamaz. | Mevcut error sözlükleri ve yeniden eğitim akışı; Simple tahmin hatası model flag'ini korur ama eski cache'i temizler. |
| ML-05 | Çözüldü: Simple importance gerçek RF feature_names_in_ ile bütün isim/ağırlıkları eşler, partial optional schema desteklenir. | Eksik/bozuk metadata mevcut boş DataFrame fallback'ine gider; Simple UI tüketimi kanıtlanmadı. |
| ML-06 | Çözüldü: Simple nonfinite return/current/derived değer veya pozitif olmayan fiyat error döndürür; HOLD/confidence üretilmez. | Taşan percent/fiyat ve eski cache izolasyonu testli. Geçerli sinyal/confidence formülü aynı, kalibre olasılık değildir. |
| ML-07 | Çözüldü: Accepted ADR-0008 tek first-interval helper ve bir kez clipping;3/14/31+ fallback yok. | Literal uç/yön/eşitlik testleri ve aynı helper'dan UI caption; format1/model aynı, finansal etkinlik kanıtı değil. |
| ML-08 | Çözüldü: fiyat sözlüğü current_price ile, legacy return kayıtları kendi kesir birimiyle okunur; invalid/mixed/error ValueError. | UI yayılım sezgisine bağlanmaz. Gözlem selector/last_index/0..h sunumu da aynı onayla kapandı; takvim sağlayıcısı eklenmedi. |

Yerel ortam notu: Windows charmap stdout, predict_price içindeki Unicode uyarıyı
yazarken hata sözlüğüne yol açtı; UTF-8 stdout ile hesaplama reproducer'ları
çalıştı. Deployment etkisi kanıtlanmadı. İki ML modülündeki process-wide warning
susturma ve genel dependency destek matrisi de bu test adımında değişmedi.

## Doğrulanmış durum — 2026-11-05

- **Önceki BL-003 hata-yolu dilimi:** Mevcut provider 8/2/1.5 ayarları değiştirilmeden history
  timeout'una iletildi. Legacy override yokluğu, gerçek adaptör zinciri ve
  1.5/3.0 backoff doğrulandı. Görünür integer HTTP >=400 RSS gövdesi haber sayılmaz;
  bozo/redirect uyumu ve kalan kaynaklar korunur. Yeni RSS policy/metric yok.
  24 hedef / 103 birleşik test geçti. Ayrıntılar `2026-11-05-bl003-failure-paths.md`.
- **Güncel karar/uygulama:** ADR-0005 Accepted; RSS connect/read 8/8, en fazla
  2 ek deneme, 1.5/3.0 bekleme, seçili retry nedenleri ve Retry-After ertelemesi
  uygulandı. RSS/Yahoo geçici kaynak tanıları ve legacy kısmi kaynak uyarısı var.
  SDK genel deadline, global rate-limit/cache, diğer Yahoo metotları ve merkezi
  üretim alarm/metric kapanışı iddia edilmez; BL-003 genel kapsamı açık.
  31 yeni testle toplam 134 dış ağ engelli test geçti (46 RSS, 18 provider/adaptör,
  13 kontrollü Streamlit dahil); önceki 10 cockpit senaryosu korunur, yeniden koşulmadı.
- **Güncel BL-002 uygulaması:** Cockpit analiz metadata'sını gösterir; sağlayıcı,
  demo ve bilinmeyen kaynakları ayırır. Analiz hatası, sembol/oturum değişimi ve
  geç gelen eski yanıt sonucu yeniden görünür yapmaz. Başlangıç/boş/veri yok/hata
  metinleri ayrıldı. On senaryo üretim frontend'inde geçti; build başarılı.
- **ADR-0004 Accepted:** RSS yayın ve çekilme zamanı ayrıdır; bilinmeyen tarih
  uydurulmaz. NLP hatası/sonlu olmayan skor `None/failed`; gerçek sıfır nötrdür.
  Özet/filtre skorsuz haberi nötr saymaz. 14 çevrimdışı sözleşme testi geçti.
- **Legacy:** 8 Streamlit AppTest gerçek fonksiyonları kontrollü RSS/ML/IO ile
  render eder. Deneysel metinler, nullable okuyucular, sabit olasılıkların
  kaldırılması ve geçersiz yayılımda senaryo/bant/uyum/risk gizleme doğrulandı.
  Tam monolit browser/canlı sağlayıcı kanıtı değildir. 82 test birleşik geçti.
- **Açık kapsam:** ML ufuk/takvim tarihi, kesin eğitim penceresi/kalibrasyon,
  sentiment etiket/filtre eşik uyumu ve genel M1/M2 kapanışı değişmedi.

### Önceki test ve inceleme kayıtları (uygulama öncesi)

- BL-001 kısmen tamamlandı: önceki 35 teste 14 patern ve 11 sinyal testi eklendi.
  Toplam 60 backend testi dış socket bağlantıları engellenerek geçiyor;
  Windows asyncio için yalnız loopback'e izin verildi. Yeni testler ağ/SMTP
  kullanmaz. Doji/FVG/RSI eşikleri ve çoğunluk kararı için dört bellek içi
  hatalı varyant yeni testlerle yakalandı; üretim dosyaları değiştirilmedi.
  BOS/kombolar, risk/pozisyon ve ML yardımcıları tamamen kapsanmadığı için
  BL-001 ve M1 açık kalır.
  Bağımsız inceleme ek açığı: MACD/Bollinger sabit seri testleri geçiyor;
  dönem/işaret/standart sapma çarpanı hatalarını yakalayacak bağımsız beklenen
  değerli değişken seri fixture'ları bu inceleme sırasında eksikti. Bu dar açık
  2026-11-06'da 5 test ve 12 mutant kontrolüyle kapatıldı; diğer BL-001 alanları açık.
- Yeni testlerin bağımsız incelemesinde engelleyici bulgu yok. Düşük öncelikli
  ek kapsam: hammer/shooting-star kısa gölgesinde kayan noktasız tam eşitlik,
  yıldızlarda ilk gap'in bağımsız bozulması, çift tepe/dipte %2 eşiğine yakın
  eşit olmayan ekstremumlar. Mevcut testler bu ek sınırları kapatmış sayılmaz.
- Deterministik UI: Beş senaryo backend olmadan sabit route yanıtlarıyla hem
  geliştirme hem üretim frontend'inde geçti. Başarı/boş/kısmi/veri yokluğu/503
  ve yeniden deneme ile açık/kapalı RSI tercih kalıcılığı koşulsuz sınanır.
  Üretim UI/API koduna mock modu eklenmedi. Bağımsız incelemede engelleyici bulgu yok.
- UI metni/hata sunumu açık: boş sonuç “Henüz tarama yapılmadı.” metnini koruyor;
  `detail.message` içeren FastAPI hatası genel “İstek hatası” olarak gösteriliyor.
  Doğru boş durum metni ve hata ayrıntısı gösterimi ayrıca ele alınmalı;
  BL-002 ürün metni onay kapısı korunur. Analiz fixture'ı tam API sözleşmesi değildir.
- BL-002 kaynak incelemesi: cockpit analiz metadata'sını ekrana taşımıyor ve
  hata halinde eski analizi koruyor. Legacy haber gerçek RSS kullanıyor; mock
  yardımcılar için diğer Python modüllerinde doğrudan çağrı bulunmadı. ML uyum
  skoru ve grafiği başarı olasılığı/güven aralığı diye doğrulanmış değildir.
  Kesin metinler A diliminde Proposed. RSS yayın tarihinin bugünün saatiyle
  doldurulması ve NLP hatasının nötr `0.0` sayılması için ADR-0004 önerildi;
  kabul edilmiş karar veya uygulanmış düzeltme değildir.
- BL-002 bağımsız incelemesi ML senaryo kartlarının sabit %30/%40/%30 olasılık
  iddialarını ek bulgu olarak belirledi; kaldırılmaları ve geçersiz yayılım
  fallback'inin görünür sonuç yerine hata olarak sunulması A önerisine alındı.
  Model tahmin ufku/takvim tarihi ve sentiment etiket/filtre eşik tutarsızlıkları
  ayrıca açık kalır; hesaplamalar sessizce değiştirilmez.
- BL-008 satırındaki `Accepted`, yalnızca ADR-0002'deki mimari hedefi ifade eder;
  veri lisansı, privacy ve operasyon onaylarının tamamlandığı anlamına gelmez.
- BL-009 hedefi kabul edilmiştir; mevcut yerel API'nin çalışması PostgreSQL,
  snapshot worker, backup veya üretim dağıtımının tamamlandığına kanıt değildir.

## İlk beş iş — tarihsel uygulama sırası

Bu ilk listenin dar dilimleri aşağıda kayıtlıdır. Güncel devam sırası ve genel
M1 kapanışının kalan işleri [M1_CLOSEOUT](M1_CLOSEOUT.md) kaydındadır.

| Sıra | İş / dosyalar | Kabul ve doğrulama | Durum / kapı |
|---|---|---|---|
| 1 | Gerçek hesaplayıcı regresyonları: `backend/tests/test_technical_analysis.py`, `backend/tests/test_api_contracts.py` | EMA/SMA/VWMA, RSI yön/karma seri/21–22 bar sınırı, MACD/Bollinger, destek-direnç, giriş verisinin korunması; API'de gerçek hesaplama ve kısa veri uyarısı ağsız geçer. `make test-backend` | Bu dar adım tamamlandı; BL-001 bütünü değil |
| 2 | Patern ve sinyal testleri: `backend/tests/test_pattern_recognition.py`, `backend/tests/test_alert_signals.py` | Sentetik tepe/dip ve eşik-altı/üstü girdilerle olumlu/olumsuz sonuçlar bağımsız beklenen değerlerle doğrulanır; ağ/SMTP çağrısı yapılmaz. | Dar adım tamamlandı (25 yeni test); finansal davranış değişmedi, BL-001 bütünü değil |
| 3 | Deterministik tarayıcı senaryoları: `scripts/run_frontend_ui_smoke.mjs`, `scripts/frontend_ui_scenarios.mjs` | Teste özel kontrollü API yanıtlarıyla başarı/boş sonuç/veri yokluğu ve RSI tercih kalıcılığı koşulsuz sınanır; yanıt bekleyicileri tıklamadan önce kurulur; canlı smoke ayrı raporlanır. | Dar adım tamamlandı; 5/5 senaryo dev/üretim ekranlarında geçti, üretim kodunda mock modu yok |
| 4 | Veri niteliği görünürlüğü: `frontend/app/page.js`, legacy sentiment/ML ekranları | Her ilgili ekranda gerçek/mock/deneysel ayrımı ile kaynak ve zaman görünür; hatada sentetik fiyat gerçekmiş gibi sunulmaz; ekran kontrolleri eklenir. | A ve ADR-0004 onaylı dilimi uygulandı; 14 RSS/NLP, 8 kontrollü Streamlit, 10 cockpit senaryosu. Genel BL-002/M1 kapanışı açık |
| 5 | Entegrasyon hata yolları: `backend/app/providers/market_data.py`, `modules/data_fetcher.py`, `modules/sentiment_analyzer.py` | Başarı, timeout, retry tükenmesi ve kısmi kesinti kontrollü testlerle doğrulanır; mevcut `timeout_seconds` alanı history çağrısına iletilir. | Dar düzeltme ve Accepted ADR-0005 RSS bütçe/tanı/legacy uyarı dilimi uygulandı; genel BL-003/M1 açık |

M2'nin alarm/snapshot akışları bu listenin bitmesiyle otomatik kapanmaz.
Kalıcı şema/migration, üretim kimlik doğrulama, SMTP/vendor, deployment ve maliyet
değişiklikleri için AGENTS.md onay kapıları geçerlidir. Constitution hâlâ Draft.

## M1 sonrasında kalan aşamalar ve kapanış ölçütleri

Bu tablo mevcut ürün/ADR hedeflerini toplar; teknik tasarımı veya açık ürün
kararını onaylanmış saymaz. Başlamadan önce ilgili ayrıntılı plan netleştirilir.

| Aşama | Kalan iş | Kapanış kanıtı / onay |
|---|---|---|
| M1 / BL-001 | ML yardımcıları ve kalan ek patern/yapı kenarları | RP-01–05 bilinen/onaylı kapsamı düzeltildi; ADR-0006 OTT gerçek zinciri testli. Genel M1 kapanışı değildir; yeni veri/sürüm/algoritma kararları ayrıca değerlendirilir. |
| M2 / davetli erişim | Demo token akışından gerçek davet + e-posta link operasyonuna geçiş | Davetli/davetsiz, kullanılmış/süresi dolmuş link ve oturum senaryoları doğrulanır; güvenlik/gizlilik, SMTP/vendor ve operasyon sahipliği açık onay ister. |
| M2 / cockpit ve paper | Auth/hata/boş veri/yeniden deneme, watchlist ve paper akışlarının kapanışı; desteklenen sembol kümesinin doğrulanması | Temel akışlar kontrollü API/tarayıcı testleriyle geçer; BIST100 listesinin güncelleme kaynağı ve sorumlusu belirlenir. Gerçek emir veya muhasebe eklenmez. |
| M2 / alarm ve snapshot | In-app + SMTP bildirimleri ve ADR-0002 hedefindeki günlük snapshot iş hattı | Başarı/hata/tekrar çalıştırma, idempotency ve veri tazeliği kanıtı; worker/queue veri modeli, saklama ve servis ayarları uygulamadan önce onaylanır. |
| M3 / kalıcılık | Auth/watchlist/paper ve iş kayıtları için PostgreSQL geçiş tasarımı | Şema/migration ve veri saklama onayı; geçiş ve geri dönüş testleri; süreç yeniden başlayınca korunması gereken kayıtlar korunur. |
| M3 / dağıtım | Hostinger/Coolify API/web/worker ortamları ve sürüm akışı | Maliyet/vendor, secrets, veri lisansı ve operasyon sahibi onayı; staging ana akış, health, deploy ve rollback denemeleri. |
| M3 / kurtarma ve gözlem | Backup/retention, izole restore testi, sağlayıcı hata/gecikme görünürlüğü, runbook | Restore ve rollback kanıtları kaydedilir; veri sahibi, saklama/şifreleme kararları, alarm eşikleri ve sorumluları onaylanır. |
| CDSK / BL-004–007 | README placeholder'ları, desteklenen Python/Node ve dependency politikası, legacy modülerleştirme planı, izlenen artifact/veri sahipliği | Belgeler kodla eşleştirilir; destek matrisi ve gerekirse ADR onayı alınır; kullanıcı verisi veya artifact'ı onaysız silinmez. |
| CDSK / yönetişim | Constitution TBD'leri, ürün başarı ölçütleri ve açık kararlar | Ürün sahibi açık onayı; gerekli Proposed ADR'ler hazırlanır. Eski belgelerdeki hedef/mevcut durum çelişkileri sessizce kapatılmaz. |
| Git / teslim | Bekleyen değişikliklerin gözden geçirilmesi ve sürümlenmesi | Kullanıcı commit/push onayı verdi; bu teslim için dosya kapsamı ve temel secret desenleri kontrol edilir, push sonrası uzak ref doğrulanır. Bu işlem ürün kilometre taşlarını kapatmaz. |

Bilinen belge borcu: operations `RUNBOOK.md` API başlangıcını hâlâ hedef olarak
anlatıyor. `OBSERVABILITY.md` health endpoint ifadesi BL-003 diliminde mevcut
process-health / hedef dış-kaynak tanısı olarak düzeltildi.
`RELEASE_PLAN.md` yalnızca başlangıç belge teslimini anlatıyor. Üretim hazırlığı
öncesinde bunlar mevcut/hedef ayrımıyla güncellenmelidir; bugün üretime hazır
kanıtı olarak kullanılmamalıdır. Bağımlılık güvenlik riskleri ve LibreSSL
uyarısı ayrıca doğrulanıp ele alınmalıdır; bu kayıt yeni güvenlik taraması değildir.
