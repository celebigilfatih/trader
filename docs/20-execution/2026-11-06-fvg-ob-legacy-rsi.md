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

- [ ] Gerçek RSI inclusive bounds/0/100/35/65 etiketleri ve artan sıralama;
  interval-period sözlüğü+fallback,20/21/22 bar ve None/empty korunması;
  bir sembol hatasından sonra gerçek sonucun gelmesi. Yeni üretim mantığı gerekmezse eklenmez.
- [ ] Actual button AppTest gerçek taramadan sonuç DataFrame'i ve gerçek no-match
  bilgi metnini gösterir. Provider kesintisi/no-match ve FVG/OB API tanısı ayrı
  geçici probe ile raporlanır; bozuk sözleşme normatif passing test yapılmaz.
- [ ] Dış socket engelli hedef/tam backend; anlamlı transient mutantlar,
  compile/39 CDSK dosyası/11 Boot/10 Accepted ADR/link/diff; upstream validator ayrı.
- [ ] Tek fresh-context bağımsız son inceleme, gereken TDD düzeltmesi ve kalıcı
  sonuç/karar/sınır kayıtları. Önceki yetkiyle son commit/push ve remote SHA/clean tree.

## Evidence

Başlangıç `8c554ae`, temiz çalışma ağacı; önceki tam412 test tarihsel başlangıçtır.
Bu kayıtta test sayıları ve karşı örnekler çalıştırıldıktan sonra güncellenecektir.

Task1:10 yeni test ilk koşuda6 assertion failure/0 error verdi; hesaplayıcı
çıktısı doğruyken actual chart'ta rectangle yoktu. Mevcut renk/5–10 bar width ve
filled filtrelememe niyeti korunarak yalnız DataFrame okuyucu bağlandı.
Sonra yeni10 + önceki patern/kombo26 = **36/36 /2.294 s /OK**, dış socket engelli.
İki yönde literal date/zone/OHLC/size/strength ve config/short/empty/caller
koruma; seçilmemiş/no-pattern, end clamp, karışık ve filled chart testleri var.
