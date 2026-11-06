# ADR-0010 implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` task by task; verification uses checkboxes.

**Goal:** Onaylanan A paketiyle VST-01–03'ü ve eksik VWEMA kartını düzeltmek.
**Architecture:** İki mevcut analyzer helper'ı, gerekli private validasyon ve
tek legacy kart guard'ı; gerçek API/sinyal/AppTest okuyucuları.
**Tech Stack:** Mevcut Python/unittest/numpy/pandas/ta/FastAPI/Streamlit.
**Spec:** [Accepted ADR-0010](../50-decisions/ADR-0010.md).

## Global constraints

- Kullanıcı “Onaylıyorum ve devam etg” dedi; yalnız A paketi uygulanır.
- API adları/default config, uzun seri Supertrend/Wilder+fallback, skor ağırlığı,
  format1 ve diğer indikatör politikaları korunur. Kullanıcı artifact'ı okunmaz/migrate/silinmez.
- VWEMA ilk pozitif seed/N-D matematiği; sıfırda durum azalırken çıktı taşınır;
  Supertrend ilk hazır konum p; invalid atomik ValueError, boş/short bilinçli NaN.
- Aynı onaylı codex checkout/dalında native devam; önceki commit/push yetkisi sürer.
- Test koşuları yeni süreçlerdedir; çalışan kullanıcı servisi/cache'i izinsiz değiştirilmez.

## Review focus

- Rasyonel volume sensitivity/prefix/zero-gap/leading-zero/all-zero/p1/underflow ve taşmaya dayanıklılık.
- Empty/short öncesi strict config/fiyat/hacim validasyonu ve eski Series/caller korunması.
- Supertrend0..p+1 readiness; gerçek risk/sinyal/ayı/API okuyucuları; uzun seri regresyonları.
- Eksik/empty/nonfinite VWEMA kartında —/Hesaplanamadı ve nötr durum; finite Üzeri/Altı aynı.
- Yeni model formatı/otomatik eski semantik tespiti/kanonik trailing veya genel M1 kapanışı iddiası yok.

## Task 1: Matematik, girdi ve gerçek API/readers

**Files:** `modules/technical_analysis.py`, yeni `backend/tests/test_vst_contract.py`,
`backend/tests/test_api_contracts.py`; ADR/index ve bu execution kaydı.
**Interfaces:** `add_indicator`, aynı float Series alanları, scalar/null API.

- [x] Testleri önce yaz: ADR rasyonel oracles; kısa/full prefix; leading/all-zero;
  p1/zero-decay/finite extremes; empty; invalid bool/string/complex/NaN/inf/domain/config
  ve atomik hata. Supertrend n0/1/9/10/11, gerçek sinyal/pozisyon/ayı okuyucuları,
  weighted/null API. Yanlış exception türü assertion failure sayılır.
- [x] `.venv/Scripts/python.exe -B -m unittest backend.tests.test_vst_contract -v`
  RED çıktısını incele; normal API hedefi gerçek TestClient ile ayrı sınanır.
- [x] İki helper'ı dar sözleşmeyle uygula; eşdeğer ölçekli VWEMA kullanılarak
  fiyat*hacim ara taşması önlenebilir. Validasyon ve sonuç kontrolü yazmadan önce gerçekleşir.
- [x] Core/API/önceki geçerli testler GREEN; source kapsamını incele ve Task1 commit'i.

## Task 2: Eksik kart ve paket doğrulaması

**Files:** `app.py`, yeni `backend/tests/test_vst_legacy_card.py`; ilgili mimari,
kalite, backlog/kapanış, CHANGELOG ve Boot handoff.
**Interfaces:** Task1'in same-index VWEMA Series'i; scalar çıkarma ve altıncı
kartın gerçek AST bloğu kontrollü Streamlit AppTest içinde çalıştırılır.

- [x] Mevcut kartı actual source assignment/with bloğuyla render et: None/NaN/
  ±inf, boş/missing seri, gerçek all-zero-volume helper ve finite Üzeri/Altı.
  Eksik değerlerde RED —/Hesaplanamadı; tüm app/browser diye sunulmaz.
- [x] Yalnız VWEMA5 okuma/default ve card guard'ını düzelt; aynı AppTest GREEN.
- [x] Core/API/AppTest ve tam backend dış socket engelli; uygun mutantlar,
  compile/CDSK yapı/link/ADR/diff. Mevcut geçerli13 hesaplama testi aynı kalır.
- [x] Bağımsız bütün değişiklik incelemesi; bulguların gereken TDD düzeltmesi;
  belgeler/Boot güncellemesi.
- Teslim: önceki açık yetkiyle aynı dala commit/push; remote SHA ve clean tree
  son yanıtta doğrulanır. Nihai teslim SHA'sı git log/remote kaydıdır.

## Evidence

Task1: yeni19 core testi önce114 assertion failure/0 error; üç yeni gerçek
API testi önce3 failure/0 error verdi. Uygulama sonrası core19 + önceki12
hesaplama testi31/31 (0.190 s), gerçek API18/18 (0.278 s), dış ağ engelli.
API'de ilk39 sabit100 sonrası130/hacim1000 sonucu VWEMA5=21700/167,
VWEMA20=87300/673; all-zero hacim null, Supertrend9/10 null ve11 barda94.
Yalnız helper'ların empty/tek bar sözleşmesi bu pakettedir; mevcut genel
sinyal/pozisyon okuyucularının en az iki fiyat gereksinimi genişletilmedi.

Task2: dört actual-card AppTest RED7 assertion failure/0 error (None render
hatası assertion olarak raporlanır), dar guard sonrası4/4 (2.226 s).
Eksik/empty/missing key/NaN/±inf ve gerçek all-zero VWEMA akışı —/Hesaplanamadı;
finite fiyat/Üzeri/Altı renkleri korunur. Tam **412 test /32.881 s /OK**,
dış socket engelli, Windows asyncio loopback serbest. Önceki geçerli13 test aynı.

15 bellek içi geçerli mutant: hacmi yok sayma, zero-decay atlama, sahte sıfır,
gelecek seed, alpha/önceki mass, zero bar fiyatına atlama; Supertrend erken seed,
warmup yönü/bfill, H<L/overflow denetimi; sıfır period, bool/negatif fiyat.
Hepsi assertion failure/0 error ile yakalandı. İlk harness anchor uyumsuzluğu
mutant sonucu sayılmadı; diskte üretim mutasyonu yapılmadı.

Operasyon sınırı: testler yeni süreçlerde çalıştı; kullanıcı cache/servisi ve
artifact'ı değiştirilmedi. Optional eski bu özellikli modeller yeniden eğitim
ister; format1/CI/config/dependency aynı. Frontend production build/browser,
canlı provider ve uzak CI bu backend/legacy diliminde yeniden koşulmadı.
Upstream CDSK validator mevcut iki bilinen konumda yok; yapı kontrolü onun
yerine geçen bir başarı olarak sunulmaz. VST-01–03 dar dilimi kapandı; genel M1,
FVG/OB aktif bağlantıları, legacy StockScreener ve finansal etkinlik açık.

Restore hedef53/53 (4.287 s).67 Python kaynağı compile,39 zorunlu CDSK dosyası,
11 Boot bölümü,10 Accepted ADR ve91 değişen-belge yerel bağlantısı doğrulandı;
git diff --check temiz. Bunlar upstream validator çalıştırıldığı anlamına gelmez.

## Bağımsız son inceleme ve sınır kararları

Fresh-context inceleme b990d44..d88ff30 ve iki tarihsel belge netleştirmesini
kontrol etti; Critical/Important/Minor bulgu yok. İncelemecinin kendi koşusu
**53/53 /2.305 s /OK**, dış socket engelli. Son görev kapısı tam paketi yeniden
**412/412 /16.842 s /OK** doğruladı. Encoded log için helper grep'i binary match
yazdı; gerçek log sonundaki test sayısı/OK ayrıca okunup doğrulandı.

İncelemecinin kapsam dışında bıraktıkları için yürütücü kararları:

1. Finansal etkinlik/canonical trailing doğrulanmış sayılmaz; onay mevcut temel
   bant hesabını korur. Yanlış yorumlanırsa kullanıcı kanıtlanmamış yatırım
   başarısı veya başka bir algoritma uygulandığı varsayımına düşebilir.
2. Genel M1 kapatılmaz; yalnız VST-01–03 dilimi kapanır. Aksi karar kalan aktif
   bağlantı/provenance/operasyon boşluklarını üretime hazırmış gibi gizler.
3. Format1 ve otomatik eski semantik tespiti genişletilmez; onay bu değişikliği
   içermez. Eski bu özellikli optional modeller yeniden eğitilmezse eski
   semantiğe dayalı tahmin riski kalır; hiçbir kullanıcı artifact'ı açılmadı.
4. Canlı sağlayıcı/full-browser/uzak CI/upstream CDSK başarısı iddia edilmez;
   bu çalışma kontrollü backend/kart kanıtıdır. Bu ortamlara özel entegrasyon
   kusurları hâlâ mümkün; upstream validator bilinen konumlarda yoktur.
5. Tam412 koşusu yürütücünün doğrudan kanıtıdır; bağımsız incelemecinin tekrar
   kapsamı53'tür. Tam pakette ikinci bağımsız koşu varmış gibi sunulmaz;
   tüm test sonuçlarının bağımsız tekrar güvencesi yoktur.

Ertelenen Minor bulgu yok. Sonraki adım FVG/OB gerçek aktif okuyucu ve legacy
tarayıcı bağlantılarını mevcut sözleşmeyle doğrulamak; genel M1 açık kalır.
