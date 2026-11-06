# VWEMA and Supertrend tests implementation plan

> **For agentic workers:** Use `superpowers:executing-plans`; checkboxes track verification.

**Goal:** M1-P0-01 VWEMA/Supertrend geçerli hesaplama dilimini ve gerçek API/sinyal okuyucularını ağsız sınamak.

**Architecture:** Gerçek analyzer/ta/pandas; literal/rasyonel oracles. Yalnız
Supertrend exception fallback'i için ATR sınırında kontrollü hata enjekte edilir.

**Tech Stack:** Mevcut unittest/pandas/ta/FastAPI; yeni bağımlılık yok.

**Spec:** [M1_CLOSEOUT](M1_CLOSEOUT.md) P0-01; mevcut config ve gerçek okuyucular.

## Global constraints

- Üretim/formül/config/veri modeli değiştirilmez; yeni ADR kararı alınmaz.
- VWEMA testlerinde başlangıç penceresi sabit fiyat/hacimdir: bilinen seed
  lookahead ve hacim tanımı açığı doğru beklenti olarak sabitlenmez.
- Supertrend yeterli verideki mevcut temel bant/yön hesabıdır; canonical
  trailing Supertrend veya finansal etkinlik kanıtı değildir. Kısa veri kusuru passing test yapılmaz.
- Aynı codex checkout/dalında devam; kullanıcı değişiklikleri ve önceki teslimler korunur.

## Review focus

- VWEMA5/20 alpha ve gerçek sıfır hacimde taşıma; yalnız sabit başlangıçtan sonraki aritmetik.
- Gerçek ATR/Wilder ile exception fallback rolling-mean'in ayrı literal beklentileri.
- Supertrend HL2≠Close, gap true range, iki yön dönüşü ve tam eşitlikte mevcut <= sınırı.
- Gerçek generate_signal ve API okuyucuları; hesaplayıcı double'ı kullanılmaması.
- VST-01/02/03 karşı örnekleri kapsam kapanışı veya normatif passing oracle sayılmaması.

## Task 1: Hesaplama ve okuyucu testleri

**Files:** Yeni `backend/tests/test_vwema_supertrend.py`; `test_api_contracts.py`
içinde bir gerçek hesaplama testi; execution/backlog/kapanış/milestone/kalite/Boot/CHANGELOG.

**Interfaces:** Mevcut add_indicator/get_latest, AlertSystem.generate_signal ve
analiz API'sinin vwema_5/vwema_20 scalar alanları; yeni public arayüz yok.

- [x] VWEMA5 için5 sabit100 ardından130/80/160 →110/100/120;
  VWEMA20 için20 sabit100 ardından121/81/142 →102/100/104. Hacim1.
  Aradaki sıfır hacimde fiyat atlaması yok; config period2 ile120/100/130.
- [x] Supertrend10/mult3 için10 adet H101/L99/C100, ardından H/L/C
  (114,108,110),(94,88,90),(124,118,120),(104,98,100).
  Gerçek ATR çıktısında fiyat101.4/106.24/97.084/72.8756, yön1/-1/1/1.
  ATR hatasında gerçek manuel fallback fiyat101.4/106.6/95.8/69.8.
  Period2/mult1 fiyat103/106/96.5/77.75; equality örneği94'e eşitte yön-1.
- [x] Girdi/index korunması ve sabit ortak seed sonrası VWEMA prefix;
  gerçek generate_signal prefix11..14 →BEKLE/SAT/AL/BEKLE.
  API'de37 sabit100 ardından121/81/142 ile vwema5=1016/9,vwema20=104.
- [x] Hedef `python -B -m unittest backend.tests.test_vwema_supertrend backend.tests.test_api_contracts -v`
  kurulu `.venv/Scripts/python.exe` ile dış socket engelli çalışır. Expected: yeni12 core+1 API ve mevcut14 API geçer.
- [x] Geçerli transient alpha/period/volume-branch/ATR/HL2/band/yön/okuyucu
  mutantlarını assertion failure ile yakala; restore green. Bilinen seed/volume tanımı kusurları kill kanıtı sayılmaz.
- [x] Tam backend dış socket engelli; compile/CDSK yapı/link/diff; bağımsız
  son inceleme, belgeler ve Boot handoff'u. Expected: tüm testler OK; M1 açık.

**Teslim:** Mevcut açık commit/push yetkisiyle aynı dal; remote SHA/clean tree kontrolü.

## Açık bulgular — ayrı karşı örnekler

| ID | Yeniden üretilen davranış | Sınır / sonraki iş |
|---|---|---|
| VST-01 | VWEMA5 dört100 barında bar0=100; beşinci200 barı eklenince bar0=120. | Seed ilk pencereyi bar0'a yerleştiriyor. Causal seed/warmup sözleşmesi netleşmeli; bu geçmiş yeniden yazımı passing test yapılmaz. |
| VST-02 | Beş sabit100'den sonraki130 barının hacmi1 veya1000 iken sonuç110. | Pozitif hacim Close*V/V'de sadeleşiyor; başlangıç sonrası hacim ağırlığı tanımı açık. Finansal formül seçimi ayrıca kararlaştırılmalı. |
| VST-03 | Supertrend9 barda fiyat yok/yön+1;10 barda bar9=94,11 bar eklenince bar9 NaN. | Kısa veri readiness ve seed konumu tutarsız. Fiyatı olmayan yön risk/pozisyon okuyucusuna açık; yeni girdi/warmup politikası ayrı karar gerektirebilir. |

VST-03 kontrollü9 bar ile gerçek `generate_position_recommendation` okuyucusunda
da yeniden üretildi: Supertrend eklenmeden bull_score0; fiyatı olmayan yönle
bull_score1.5 ve “SuperTrend Pozitif”. Bu bir passing beklenti değildir.
Mevcut BIST fetcher en az15 bar ister; bu karşı örnek doğrudan çekirdek
okuyucusuna kısa veri verilmesidir, canlı sağlayıcı/UI akışında aynı girdinin
geldiği iddiası değildir. VST-01/02 ise yeterli uzunluktaki VWEMA hesabını da ilgilendirir.

## Evidence

- 12 yeni core+1 gerçek API testi eklendi. Yeni core12/12; API15 ile
  **27/27 /0.289 s /OK**. Gerçek generate_signal ve analiz route okuyucuları
  çalışır; chart/browser veya bütün risk/pozisyon okuyucuları doğrulanmış sayılmaz.
- 32 geçerli bellek içi alpha/period/volume-branch/ATR/HL2/bant/yön/okuyucu
  mutantı assertion failure/0 error verdi; restore27/27 (0.443 s).
  Bilinen seed/pozitif hacim tanımı kusurları mutant kill kanıtı yapılmadı.
- Tam **386 backend testi /28.692 s /OK**, dış socket engelli, Windows asyncio
  loopback serbest. AppTest/TestClient gereksinimiyle sandbox dışında koşuldu.
- Üretim/formül/config/ADR/CI/dependency değişmedi. Frontend/canlı sağlayıcı
  ve uzak CI yeniden koşulmadı; upstream CDSK validator yokluğu ayrı kalır.
- 65 Python compile/39 zorunlu CDSK dosyası/11 Boot/9 Accepted ADR ve yerel
  belge bağlantıları/diff kontrolü geçti. Bu upstream validator başarısı değildir.
- VST-01–03 açık: sonraki iş bu başlangıç/hacim/kısa veri davranışlarını dar
  sözleşme/düzeltme kararına dönüştürmek; FVG/OB aktif bağlantıları ve legacy
  StockScreener kapsamı da genel M1 için açık. Yeni politika sessizce uygulanmadı.
- Son inceleme eyleme dönük bulgusuz: reviewer ilk hedef27/27'yi0.262 s'de,
  sıfır hacim kapsamı daraltıldıktan sonra güncel core12/12'yi0.036 s'de doğruladı.
  Sıfır hacim testi yalnız taşıma noktasında biter; sonraki pozitif hacimde
  çözümlenmemiş ağırlık formülünü sabitlemez. Bu son değişiklikten sonra controller
  hedef27/27,32 mutant/0 error ve tam386 paketi yeniden çalıştırdı.
  Reviewer tam paket/mutant/dış sistemleri veya finansal etkinliği bağımsız tekrar doğrulamadı.
