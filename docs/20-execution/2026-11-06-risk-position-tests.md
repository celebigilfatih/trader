# BL-001 risk ve pozisyon test planı

> Bu kayıt düzeltme öncesi test/reproducer aşamasını anlatır. Sonraki RP
> düzeltmeleri ve kalan OTT onay kapısı `2026-11-06-risk-position-fixes.md`
> içindedir; aşağıdaki açık/üretim değişmedi ifadeleri o önceki aşamaya aittir.

**Goal:** Mevcut risk skoru, pozisyon puanlaması ve yerel DayTrader hesaplarını
deterministik testlerle doğrulamak; yeni finansal politika seçmemek.
**Architecture:** Gerçek AlertSystem, TechnicalAnalyzer ve DayTrader çalışır.
Kural sınırı testlerinde açık sentetik indicator serileri kullanılır; gerçek
indikatör zinciri örnekleri ayrıca bulunur. Ağ/SMTP/gerçek emir yoktur.
**Tech Stack:** Mevcut unittest/pandas/numpy ve standart test araçları.
**Spec:** BACKLOG BL-001 ve TEST_STRATEGY P0 çekirdek kapsamı.

## Ortak sınırlar / review odağı

- Başlangıç HEAD `6ed1a59`, aynı dal. Önceki değişiklikler korunur; commit/push,
  üretim formülü/config/API/model/dependency değişikliği kapsam dışıdır.
- Sonuçlar prototipin sezgisel puanları ve paper/bellek kayıtlarıdır; yatırım
  tavsiyesi, başarı olasılığı, üretime hazırlık veya gerçek muhasebe kanıtı değil.
- Beklentiler gerçek çıktıdan türetilmez: sabit girdiler ve elle kontrol edilmiş
  sayılar. Mevcut davranış testleri gerçekçi bellek içi mutantlarla sınanır;
  eksik feature için RED üretildiği iddia edilmez. Diskte mutation yapılmaz.
- İnceleme odağı: eşitlik/alt/üst eşikleri, yön, toplama ve ağırlıklar, yıllık
  volatilite/sample std, hacim ortalamasına son barın katılması, ATR ve entry
  offset'i, stop/hedef/R:R, miktar/yön bazlı P&L ve kayıtların doğru taşınması.
- Sıfır/nonfinite göstergeler, boş/kısa veri, geçersiz position type/size veya
  desteklenmeyen parametre için yeni politika seçilmez. Açık hatalar yeniden
  üretilip kaydedilir; riskli davranış başarılı regresyon beklentisiyle onaylanmaz.

## Task 1 — Birleşik risk (ana ajan)

**Create:** `backend/tests/test_risk_analysis.py`.

- [x] Gerçek analyzer ve açık kural girdileriyle RSI/MACD/trend/hacim/destek-
  direnç eşikleri, volatilite sınıfları ve son 20 getiri/20 getiri warmup'u.
- [x] Toplam skor 2/4/7 sınırları ve altları; pozisyon yüzdesi, hesaplanmış
  stop/hedef tutarları ve 100 dışı fiyat/yuvarlama. Gerçek indikatör zinciri,
  girdi korunması ve tekrar çağrıda bağımsız sonuçlar.
- [x] Testler geçer; toplama/yön/eşik/pencere/çarpan mutantları assertion ile
  yakalanır. RSI 0/MACD 0 truthiness şüphesi ayrı reproducer ile incelenir.

## Task 2 — Pozisyon öneri kuralları (pozisyon worker)

**Create:** `backend/tests/test_position_recommendation.py`.

- [x] Gerçek AlertSystem.generate_position_recommendation; RSI/MACD/EMA/MA200/
  SuperTrend/OTT/BB/VWAP/hacim katkıları ve toplam karar/confidence sınırları.
- [x] İki yön, TUT, eşitlik ve karışık sinyal/az hacim uyarıları, input korunması;
  ayrıca gerçek indikatör zinciri. Şüpheli güçlü SAT sizing'i reproducer ile
  raporla; beklenen doğru davranış gibi dondurma veya sessizce düzeltme.
- [x] Hedef testler ve geçerli transient mutantlar; mutation sonrası green.

## Task 3 — DayTrader giriş/çıkış ve yerel kayıtlar (DayTrader worker)

**Create:** `backend/tests/test_day_trader_risk.py`.

- [x] Gerçek ATR gap bileşenleri/rolling ve entry/stop/hedef/R:R örnekleri;
  BUY/SELL/HOLD ve entry offset etkisi, gerçek analyzer zinciri, yuvarlama.
- [x] Yerel BUY/SELL P&L yön/miktar, aktiften geçmişe taşıma, kapanma zamanı/
  sebebi ve olmayan sembol etkisizliği; kullanıcı verisi veya disk yazma yok.
- [x] Hedef testler ve gerçekçi transient mutantlar; gerçek kod geri green.
  Kısa/boş veri ve beklenmeyen branch bulguları ayrı, yeni politika yok.

## Task 4 — Birleşim ve kapanış

- [x] Tüm testler dış socket engelli (Windows asyncio loopback serbest); fresh
  salt-okunur reviewer yeni dosyaları/elle hesapları ve iddia sınırlarını inceler.
- [x] Python compile, CDSK 39 zorunlu belge/11 Boot bölümü, link/ADR/diff ve
  üretim/manifest değişmezliği kontrolü. Upstream validator yokluğu ayrı yazılır.
- [x] BACKLOG, TEST_STRATEGY, OVERVIEW, CHANGELOG, PROJECT_BOOT ve bu kayıt
  güncellenir. Önceki 166 test/10 cockpit senaryosu başlangıç kanıtıdır.
  Genel BL-001/M1 açık; frontend/canlı servis yeniden koşulmaz.

## Kanıt ve bulgular

- Risk 13 test: nötr/RSI/MACD/EMA/hacim/destek-direncin katkıları, yıllık sample
  volatilite, 20 getiri warmup/penceresi, 2/4/7 toplam skor ve altları, gerçek
  indikatör zinciri, fiyat tutar/yuvarlama ve sonuç/girdi izolasyonu geçti.
  Alternatif getiriler için sample variance `5*(a-b)^2/19`; yıllık std
  `abs(a-b)*sqrt(1260/19)`. Tek %4 getiri + 19 sıfır için yıllık %14.2.
  Stop/hedef 123.456 fiyatında 114.81/149.38; risk tier 7 fiyat 100'de 98/103.
- İlk `tail(19)` getiri mutantı dengeli alternatif seride kaçtı: 19 ve 20 örneğin
  sample std'si bu özel seride aynıydı. Asimetrik tek şok beklentisi eklendi;
  tekrar tam turda 32/32 geçerli risk mutantı assertion failure, error sıfır.
  Toplama/ağırlıklar, kategori ve strict eşikler, std/yıllık katsayı/pencere,
  hacim ortalaması, tier/pozisyon/stop/reward hataları yakalandı. Gerçek 13 green.
- Pozisyon 16 test: tüm katkılar, iki yönlü toplam/karar/renk/confidence,
  gross-score uyarıları, geçerli sizing dalları ve girdi/list bağımsızlığı geçti.
  Gerçek 240 fiyatlık zincir boğa 8.5 - ayı 2 = 6.5 puandır. OTT açık trend serisiyle
  tüketici sınırında test edilir; gerçek OTT runtime kanıtı değildir.
  İlk `tail(19)` hacim mutantı ek yakın-eşik fixture'ıyla yakalandı. Nihai 38/38
  geçerli mutant assertion failure, error sıfır; gerçek 16 tekrar green.
- DayTrader 17 test: gerçek BUY/SELL/HOLD zincirleri, ATR/volume/entry offset ve
  R:R, yuvarlama, yerel kayıt/P&L/tarih/durum/history taşıma. Gap ATR `43/14`,
  giriş zinciri ATR `29/14`; BUY entry 101.202, stop 97.893, target 107.214,
  R:R 1.82. Offset nedeniyle R:R tam 2 değildir. Küçük fiyat fixture'ı, önce
  yuvarlayıp sonra R:R hesaplama hatasını yakalar. İki consumer testinde açık
  seri taşıyıcısı vardır; gerçek zincirler ayrıca gerçek TechnicalAnalyzer kullanır.
  Ağ/SMTP/dosya engelli, saat sabit. 34/34 geçerli mutant assertion failure,
  error sıfır; gerçek 17 tekrar green. Diskte üretim mutation yok.
- Bağımsız salt-okunur reviewer 46 hedef testi çalıştırdı, elle hesapları ve
  iddia sınırlarını doğruladı. Critical/Important/eyleme dönük minor bulgu yok.

### Açık reproducer'lar — düzeltme veya yeni politika değil

| Kayıt | Sabit girdi ve gözlenen sonuç | Sınır / sonraki değerlendirme |
|---|---|---|
| RP-01 | Risk nötr veride RSI0→skor0, RSI0.01→skor1; MACD-1/signal0→skor0, signal-.01→skor1.5. Pozisyonda RSI0→TUT, RSI.01→AL; gerçek düşen fiyatlarda hesaplanan RSI0 da atlanır. | `if value` geçerli sıfırı yok sayar. Mevcut None/NaN ayrımı ve tüketici koşulları düzeltilmeli; puan ağırlıkları yeniden seçilmez. |
| RP-02 | RSI70 + MACD-2/signal-1: ayı4/toplam-4/GÜÇLÜ SAT; position_size “Portföyün %75-100'ü”. | Genel GÜÇLÜ dalı SAT dalından önce. Fonksiyon reproducer'ı; app.py yalnız atama bulundu, bu alanın UI tüketimi kanıtlanmadı. |
| RP-03 | Gerçek `add_indicator('ott')`, pandas3.0.1: `NDFrame.fillna() got an unexpected keyword argument 'method'` TypeError. Kaynakta üretici ott/ott_signal, okuyucu ott_trend. | Runtime uyumluluğu ve sinyal sözleşmesi ayrı sorunlardır. Çıktı hesabı/seed ve yön politikası onaysız değiştirilmedi; çalışan OTT zinciri iddiası yok. |
| RP-04 | `[100]*11+[99,101]` ile doğrudan 13-bar entry: BUY101.202, NaN stop/hedef, R:R0/confidence45. | ATR14 eksik. Mevcut scan `<50` barı atlar; gerçek scan'de erişildiği iddia edilmez. Geçersiz sonuç için raise/HOLD gibi yeni karar bu dilimde alınmadı. |
| RP-05 | Boş ATR IndexError; desteklenmeyen position type, entry100/exit105/size2 kapamada SELL yolu ile -10 P&L. | Kısa/boş/geçersiz girdi sözleşmesi tanımlanmalı. Yanlış davranış passing golden teste çevrilmedi; doğrulama/fallback politikası seçilmedi. |

- Tam reproducer'lar gerçek sınıflarla yerelde, ağsız çalıştırıldı. Üretim
  dosyası/config/API/model/dependency değiştirilmedi, yeni ADR/commit/push yok.
  Bu açıklar nedeniyle tüm risk/pozisyon entegrasyonu veya genel M1 kapanmadı.
- Son birleşik doğrulama: yeni yakın-eşik/asimetrik fixture'larla unittest
  discover `backend/tests/test_*.py` 212/212 geçti. Socket connect/connect_ex dış
  hedeflerde engelli, Windows asyncio loopback serbest, create_connection engelli.
  49 Python compile, 39 zorunlu nonempty belge, 11 Boot bölümü, yerel Markdown
  bağlantıları/Accepted ADR durumları ve yeni dosya whitespace kontrolü geçti.
  `git diff --check` temiz; alert/day/technical/config ve manifestlerin HEAD'e
  göre diff'i boş. Upstream CDSK validator Windows ortamında yok; yapısal
  kontrol onun yerine geçti sayılmaz. Frontend build/UI/canlı servis koşulmadı.
