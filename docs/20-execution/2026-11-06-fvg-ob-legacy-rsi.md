# FVG/OB and legacy RSI connection implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` inline; verification uses checkboxes.

**Goal:** M1-P0-01 FVG/OB seçici→gerçek hesaplayıcı→grafik ve M1-P0-02'nin
ilk legacy RSI dilimini doğrulamak; bozuk okuyucu sözleşmelerini gizlememek.
**Architecture:** Mevcut TechnicalAnalyzer/AdvancedPatternRecognition ve
StockScreener korunur; yalnız mevcut grafik okuyucusunda DataFrame uyumu gerekebilir.
**Tech Stack:** Mevcut Python/unittest/pandas/ta/Plotly/Streamlit/FastAPI.
**Spec:** [M1 kabul matrisi ve sıralı kapsam](M1_CLOSEOUT.md#kalan-p0-işleri--uygulama-sırası),
[ürün kabul kriterleri](../00-product/PRODUCT_SPEC.md#acceptance-criteria),
[iç sözleşmeler](../10-architecture/API_CONVENTIONS.md#compatibility).

## Global constraints

- Kullanıcının “Sıradaki adım ile devam et” isteği mevcut M1 test/bağlantı
  kapsamını yetkilendirir; yeni formül/çıktı/hata/transport politikası seçilmez.
- Var olan codex dalı/checkout ve açık commit/push tercihi sürdürülür.
- Eski finansal hesap/config/anahtarlar, API response modeli ve kullanıcı
  artifact'ları korunur; yeni dependency/vendor/servis yoktur.
- FVG batch filled bilgisi ve OB continuation hesabı korunur. Grafikte mevcut
  bullish/bearish renkleri ve5/10 bar genişlikleri kullanılır; yeni filled filtreleme
  veya causal/live strateji iddiası eklenmez. Tarih mevcut DataFrame satırından okunur.
- Bilinen API500 veya kaynak kesintisini “eşleşme yok” gösterme passing oracle
  yapılmaz; karşı örnek/etki kaydı çıkarılır. Yeni HTTP/veri/hata sözleşmesi
  gerekiyorsa önce Proposed ADR ve özel onay gerekir.

## Review focus

- İki yön/konfigürasyon/empty/short girdide literal FVG/OB sarmalayıcı çıktıları,
  kullanıcı ve analyzer verisinin korunması; alt detector mock edilmez.
- Gerçek date/low/high satırından aynı fiyat panelinde rectangle, seçili değilse
  şekil yok; son bar clamp, FVG ve OB karışımı ve filled bilgisi değiştirilmez.
- Legacy RSI gerçek Wilder, inclusive filtre/sıralama/sinyal etiketleri;
  interval-period aktarımı, ilk hazır bar, yetersiz/boş veri ve bir sembol hatası.
- Actual RSI button AST'si ile gerçek StockScreener→analyzer→DataFrame/boş ekran;
  sağlayıcı sınırı dışında finansal hesaplar gerçek, ağ/sleep yok.
- Tüm kaynak kesintisi/boş eşleşme ve API multi-output okuyucu açıklıkları
  ayrıca görünür; genel M1 veya tüm legacy tarayıcı kapanışı iddia edilmez.

## Task 1: FVG/OB gerçek sarmalayıcı ve grafik

**Files:** yeni `backend/tests/test_fvg_ob_readers.py`, gerekirse `app.py` create_chart.
**Interfaces:** `add_indicator('fvg'/'order_block')` → fvg_bullish/bearish ve
ob_bullish/bearish DataFrame; `create_chart(df, analyzer, selected_indicators)` →Plotly.

- [x] Literal tarih/zone/size/filled/strength, default ve değişen threshold/lookback,
  short/empty ve caller korunması testleri. Mevcut gerçek fixtures yeniden kullanılır.
- [x] Actual create_chart AST→real Plotly testleri: iki yön, seçim kapalı, end clamp,
  birlikte seçim. Grafikte beklenen zone RED assertion failure/0 error olmalı.
- [x] Yalnız gerekli DataFrame grafik okuyucu uyumu; hesap/formül/veri anahtarı değişmez.
- [x] Hedef testler GREEN; eski patern/kombo regresyonları, source/diff ve ilk commit.

## Task 2: Legacy RSI ve bütün paket

**Files:** yeni `backend/tests/test_legacy_rsi_screener.py`; mevcut mimari/kalite,
backlog/kapanış/CHANGELOG/Boot handoff; gerekirse somut Proposed ADR.
**Interfaces:** gerçek `StockScreener` kurucusu; yalnız get_stock_data sınırına
sentetik OHLCV/None/exception enjeksiyonu. Actual RSI button AST okuyucu.

- [x] Gerçek RSI inclusive bounds/0/100/35/65 etiketleri ve artan sıralama;
  interval-period sözlüğü+fallback,20/21/22 bar ve None/empty korunması;
  bir sembol hatasından sonra gerçek sonucun gelmesi. Yeni üretim mantığı gerekmezse eklenmez.
- [x] Actual button AppTest gerçek taramadan sonuç DataFrame'i ve gerçek no-match
  bilgi metnini gösterir. Provider kesintisi/no-match ve FVG/OB API tanısı ayrı
  geçici probe ile raporlanır; bozuk sözleşme normatif passing test yapılmaz.
- [x] Dış socket engelli hedef/tam backend; anlamlı transient mutantlar,
  compile/39 CDSK dosyası/11 Boot/10 Accepted ADR/link/diff; upstream validator ayrı.
- [x] Tek fresh-context bağımsız son inceleme, gereken TDD düzeltmesi ve kalıcı
  sonuç/karar/sınır kayıtları.
- Teslim: önceki açık yetkiyle aynı dala commit/push; remote SHA/clean tree
  son yanıtta doğrulanır. Nihai SHA git log/remote kaydıdır.

## Evidence

Başlangıç `8c554ae`, temiz çalışma ağacı; önceki tam412 test tarihsel başlangıçtır.
Bu kayıtta test sayıları ve karşı örnekler çalıştırıldıktan sonra güncellenecektir.

Task1:10 yeni test ilk koşuda6 assertion failure/0 error verdi; hesaplayıcı
çıktısı doğruyken actual chart'ta rectangle yoktu. Mevcut renk/5–10 bar width ve
filled filtrelememe niyeti korunarak yalnız DataFrame okuyucu bağlandı.
Sonra yeni10 + önceki patern/kombo26 = **36/36 /2.294 s /OK**, dış socket engelli.
İki yönde literal date/zone/OHLC/size/strength ve config/short/empty/caller
koruma; seçilmemiş/no-pattern, end clamp, karışık ve filled chart testleri var.

Task2:6 gerçek RSI ve2 actual buton AppTest ilk koşuda8/8 (0.634 s). Mevcut
geçerli davranış testidir; üretim RSI değişmedi. Wilder22 +2/-1 rasyonel oracle
65.625; inclusive filtre,0/100 ve35/65 etiketleri, sıralama,7 interval/period
örneği, None/empty/20/21/22 ve tek sembol hatasından sonra gerçek sonuç doğrulandı.
Provider sınırı kontrollü; gerçek kurucu/analyzer/ta ve buton→tablo yolu kullanıldı.

Ek uzun chart örneği OB'nin10 bar genişliğini/clamp'ten bağımsız ve iki rengini
doğruladı; chart sonrası caller/analyzer ve türetilmiş frame'ler korunur.
Son hedef **45/45 /3.694 s /OK**:11 yeni FVG/OB +8 RSI +26 önceki patern/kombo.
Önceki tam koşu430/430 (18.201 s), bu son ek testten önceki kayıttır.

14 geçerli bellek içi mutant assertion failure/0 error: RSI iki inclusive uç,
sıralama,35/65 etiketleri,interval/period,22 ilk hazır; FVG threshold,OB lookback,
grafik5/10 bar genişliği,low/high ve seçim kapalı dalı. İlk sandbox Streamlit
temp-dizin ve boş satır indeksleme harness sonuçları başarı sayılmadı; hedef
assertion netleştirilip izinli ortamda tekrarlandı. Diskte üretim mutasyonu yoktur.

## Açık karşı örnekler ve somut kararlar

| Kayıt | Gerçek okuyucu kanıtı | Durum / sonraki kapı |
|---|---|---|
| OS-02 | Auth + gerçek analyzer + kontrollü40-bar provider; fvg/order_block/bos/fvg_ob_combo/fvg_bos_combo/bollinger ayrı isteklerde500, macd200. | Skaler endpoint çoklu çıktı projeksiyonu eksik. [ADR-0011](../50-decisions/ADR-0011.md) Proposed A: altı seçici mevcut invalid_indicator400 ile açıkça reddedilir; nesne/list desteği ayrı tasarım. Henüz uygulanmadı. |
| RS-01 | Gerçek RSI65.625 +80–90 filtresi ve provider TimeoutError senaryoları actual butonda aynı info/0table/0error/0exception üretir: “Belirtilen RSI aralığında hisse bulunamadı”. | Kesinti/yetersiz veri ile no-match ayrımı eksik. [ADR-0012](../50-decisions/ADR-0012.md) Proposed A: liste korunur, geçici last_rsi_diagnostics ve success/partial/failed/empty metinleri. Henüz uygulanmadı. |

500 veya yanlış no-match testte normatif beklenen çıktı yapılmadı; tanı amaçlı
probe sonuçlarıdır. API400/nesne modeli ve RSI yeni tanı/hata metni bu genel
devam talebinden onay türetilerek değiştirilmedi. Onay kapısı AGENTS veri
sözleşmesidir; test ve sıradan grafik bağlantısı için yeni onay gerektirilmedi.
Trend gücü/özet için modül dışı Python çağıranı bulunmadı; bu aktif okuyucu
envanterinde gerekçeli N/A, future API/UI kullanımı eklenirse yeniden ele alınır.

M1-P0-01'in FVG/OB→legacy chart dilimi kapandı, HTTP OS-02 açık. M1-P0-02 ilk
RSI geçerli hesap/okuyucu dilimi doğrulandı; RS-01, boğa/teknik ve haftalık/aylık
tarama yolları açık. Genel M1/finansal etkinlik kapanışı yoktur. Kullanıcı artifact'ı,
servisi, dependency, CI ve frontend değiştirilmedi; canlı sağlayıcı/full-browser,
frontend build ve uzak CI yeniden çalıştırılmadı. Upstream CDSK validator yokluğu sürer.

Son tam paket **431/431 /18.175 s /OK**, dış socket engelli, asyncio loopback
serbest.69 Python compile/39 zorunlu belge/11 Boot/10 Accepted+2 Proposed ADR ve
değişen belgelerin yerel bağlantıları/diff kontrolü geçti; upstream CDSK sonucu değildir.

## Bağımsız son inceleme ve yürütücü kararları

Fresh-context inceleme8c554ae..bd2b6a9 ve sonraki iki approval-boundary belge
netleştirmesini kontrol etti. Yeni Critical/Important veya actionable Minor yok;
önceden var olan OS-02/RS-01 **Important** olarak ayrıca doğrulandı.
İncelemecinin kendi hedef koşusu **45/45 /1.940 s /OK**, dış socket engelli;
HTTP ve actual buton karşı örneklerini de bağımsız tekrar etti. Tam431 ve14-mutant
kanıtı yürütücüye aittir, bağımsız tam tekrar iddia edilmez. Son görev kapısı
tam paketi **431/431 /18.363 s /OK** yeniden geçti. Helper encoded log'u binary
match diye etiketledi; gerçek log sonundaki sayı/OK ayrıca okunup doğrulandı.

İncelemecinin kapsam dışında bıraktıkları hakkında yürütücü kararları:

1. OS-02/RS-01 düzeltmesini öneriler onaylanmadan uygulamamak: altı seçiciyi400
   yapma ve geçici tanı/hata sunumu mevcut sözleşme kararlarıdır. Kullanıcı etkisi
   değerlendirildi ve açık tutuldu; karar beklerken HTTP500 ve yanıltıcı no-match
   riski sürer. Kullanıcının genel devamı henüz sunulmamış iki A'nın özel onayı değildir.
2. Diğer legacy tarayıcılar/provenance/genel M1 kapanmaz: bu dilim ilk RSI ve
   FVG/OB chart'tır. Aksi karar sınanmayan zincirleri tamamlanmış gibi gösterir.
3. Canonical/causal patern ve finansal etkinlik doğrulanmış sayılmaz: batch
   hesaplar ve future-bar filled bilgisi korundu. Aksi yorum kanıtlanmamış
   yatırım/causality beklentisi yaratır; burada gerçek okuyucu doğrulandı.
4. Farklı/kesilmiş DataFrame ile ayrı analyzer eşleştirme desteği eklenmez:
   mevcut iki chart çağıranı analyzer'ı çizilen frame'den kurar. Farklı tarihli
   eşleştirmede date bulunamazsa chart hata verebilir; yeni bu kullanım kapsam dışıdır.
5. Tam monolit/browser/canlı provider/frontend build/uzak CI/dependency uyumluluğu
   başarısı iddia edilmez: kontrollü hedef/tam backend ve seçili runtime kanıtı
   vardır; bu ortamlara veya başka sürümlere özel kusurlar dışlanmış değildir.
6. Upstream CDSK validator başarısı iddia edilmez: bilinen konumlarda yoktur.
   Yapı/link/compile kontrolü standardın normatif validator sonucunun yerine geçmez.

Ertelenen Minor yok. Kullanıcıya önerilen somut sonraki kararlar ADR-0011 A ve
ADR-0012 A'dır; Proposed kalırlar. İlgili approval-boundary paragrafları AGENTS'in
“veri modeli/migration” kapısını bu sözleşmelere uygulamanın yorum olduğunu açıklar;
genel test/kod bakımına hayalî onay şartı eklenmez.
