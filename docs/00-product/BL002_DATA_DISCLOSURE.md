# BL-002 — Veri niteliği ve ekran metni önerisi

- **Status:** Accepted — ürün sahibi 2026-11-05 tarihinde “devam et” yanıtıyla onayladı.
- **Date:** 2026-11-05
- **Scope:** Mevcut cockpit ve legacy haber/ML ekranlarının kaynak, zaman ve
  deneysel nitelik sunumu. Yeni ekran, sağlayıcı, model veya veri lisansı kararı yok.
- **Authority:** ADR-0001/0003 ve PRODUCT_SPEC mevcut gereksinimlerdir;
  A ekran metinleri/davranışları ve ADR-0004 B sözleşmesi kullanıcıca onaylandı.

## İnceleme sonucu ve kanıt

İnceleme kaynak koduna ve doğrudan çağrıların AST taramasına dayanır. Bu adımda
canlı haber/piyasa verisi, tarayıcı veya Streamlit ekranı çalıştırılmadı.

| Alan | Doğrulanmış mevcut durum | Eksik / uygulanacak iş |
|---|---|---|
| Cockpit analiz | API `data` yanında `meta.source/fetched_at/as_of/staleness_seconds` gönderir; UI yalnız `response.data` saklar ve JSON olarak gösterir. | Mevcut metadata korunup analizin yanında gösterilmeli. JSON içindeki `as_of`, tek başına kaynak ve çekilme zamanını karşılamaz. |
| Analiz hatası | `analyzeSymbol` hata dalı yalnız status değiştirir; önceki `analysis` temizlenmez. | Yeni istek/hata ve seçili sembol değişiminde eski analiz güncel sonuç gibi görünmemeli. Öneri: eski sonucu temizlemek. |
| RSI | Satırda kaynak, veri zamanı, çekilme zamanı ve yaşlanma zaten görünür. | `mock/demo/test-fixture` kaynaklarının demo niteliği açık değil; bilinmeyen kaynak gerçek diye sınıflandırılmamalı. |
| Cockpit uyarısı | Görünen sayfada yatırım tavsiyesi olmadığına dair metin yok. | Genel uyarı sürekli görünmeli. |
| RSI boş/hata | Başlangıç, başarılı boş sonuç ve hata sonrası aynı boş metin görünebilir; iç içe `detail.message` genel hata metnine düşer. | Başlangıç/başarılı boş/veri yok/hata ayrı sunulmalı. Backend ham exception metni doğrudan kullanıcı metni yapılmamalı. |
| Legacy haber | `show_news_feed` gerçek RSS alan `fetch_real_news` / `get_symbol_specific_news` ve `get_market_news_summary` çağırır; kartlarda kaynak ve `date` görünür. Başlık “Gerçek zamanlı” der. | Gerçek zaman garantisi kaldırılmalı; haber metni ile deneysel duygu skoru ayrı tanımlanmalı. |
| RSS tarih doğruluğu | `published` yoksa veya parse edilemezse `date = datetime.now()` yazılır. Kaynak yayın zamanı ile elde etme zamanı ayırt edilemez. | Yalnız etiket değişimiyle kapanmaz; aşağıdaki ayrı zaman sözleşmesi işi açık kalır. |
| Mock sentiment yardımcıları | Sabit/rastgele skor, haber başlığı, sosyal sayı ve geçmiş üreten beş yardımcı mevcut. Diğer Python modüllerinde doğrudan çağrıları bulunmadı. | Aktif haber ekranı “mock haber gösteriyor” diye raporlanmaz. Yardımcılar ileride görünür akışa bağlanırsa demo etiketi zorunlu olmalı; bu iş yeni mock ekran açmaz. |
| Legacy ML | `show_ai_predictions` Yahoo günlük verisiyle gerçek yerel modelleri eğitir. Başlıkta deneysel niteliği yok; “Canlı Piyasa” ve “AI Güveni” metinleri var. | Model çıktısı deneysel olarak etiketlenmeli; son bar fiyatına canlı fiyat denmemeli. |
| ML uyum/grafik | Uyum skoru modellerin tahmin standart sapmasından hesaplanır; grafik ara günleri `linspace` ile çizer ve ±standart sapmayı “Güven Bandı” diye gösterir. | Uyum olasılık/doğruluk olarak sunulmamalı; çizgi şematik, bant görsel model yayılımı olarak açıklanmalı. Hesaplama değişmez. |
| ML senaryoları | İyimser/beklenen/kötümser kartlarında `%30 / %40 / %30 Olasılık` sabit yazılmıştır; bu oranlar modelden hesaplanmaz. Geçersiz yayılımda ayrıca ±%5 fallback bulunur. | Sabit olasılık iddiaları kaldırılmalı; geçersiz model yayılımından doğrulanmış senaryo/grafik üretilmiş gibi gösterilmemeli. |

Kaynaklar:

- [Cockpit](../../frontend/app/page.js): `analyzeSymbol`, `runRequest`, RSI tablosu.
- [API](../../backend/app/routers.py): `analyze_symbol`, `rsi_screener`;
  [metadata sözleşmesi](../../backend/app/core/contract.py): `DataContract`.
- [Legacy UI](../../app.py): `show_ai_predictions`, `show_news_feed`.
- [Sentiment](../../modules/sentiment_analyzer.py): beş mock yardımcı,
  `fetch_real_news`, `_analyze_text_sentiment`, `get_market_news_summary`.

## Önerilen uygulama yaklaşımı

Mevcut sayfa düzeninde her çıktının yanında nitelik/kaynak/zaman gösterilir;
genel uyarı da görünür tutulur. Sadece genel bir sayfa uyarısı eklemek, tekil
sonucun gerçek/demo niteliğini ve veri yaşını açıklamaz. Yeni bir laboratuvar
ekranı veya servis açmak bu işin kapsamına alınmaz.

### A — Onaylanan kesin ekran metinleri

| ID / konum | Önerilen metin |
|---|---|
| G-01 / cockpit üst alanı | **“Araştırma ve karar desteği içindir. Yatırım tavsiyesi değildir.”** |
| D-01 / bilinen Yahoo kaynaklı sonuç | **“Sağlayıcı verisi — Yahoo Finance”** |
| D-02 / açık mock/demo/test-fixture kaynağı | **“Demo / sentetik veri — gerçek piyasa verisi değildir.”** |
| D-03 / bilinmeyen kaynak | **“Veri niteliği belirtilmedi”**; kaynak alanı da yoksa **“Kaynak belirtilmedi”** |
| D-04 / metadata alan başlıkları | **“Kaynak”**, **“Veri zamanı (son bar)”**, **“Çekilme zamanı”**, **“Veri yaşı (çekilme anında)”** |
| D-05 / bulunmayan metadata | **“Kaydedilmemiş”**; bugünün tarihiyle veya sıfır yaşla tamamlanmaz. |
| R-01 / henüz taranmamış | **“Henüz tarama yapılmadı.”** |
| R-02 / başarılı, uyarısız boş sonuç | **“Tarama tamamlandı; sonuç bulunamadı.”** |
| R-03 / sonuç yok, atlanan semboller var | **“Kullanılabilir tarama sonucu yok; atlanan sembolleri kontrol edin.”** |
| R-04 / tarama isteği başarısız | **“Tarama tamamlanamadı. Yeniden deneyin.”** |
| A-01 / analiz isteği başarısız | **“Analiz alınamadı. Yeniden deneyin.”** |
| N-01 / legacy haber alt başlığı | **“RSS haberleri ve deneysel metin duygu analizi”** |
| N-02 / haber duygu açıklaması | **“Duygu skoru, haber metninden hesaplanan deneysel bir sınıflandırmadır; haberin doğruluğunu veya fiyat yönünü doğrulamaz.”** |
| N-03 / haber özeti zamanı | **“Özet oluşturulma zamanı”** (`last_updated`); kaynak yayın zamanı diye gösterilmez. |
| M-01 / ML sayfa başlığı ve giriş bağlantıları | **“Deneysel AI Tahminleri”** |
| M-02 / ML sayfası üst uyarısı | **“Deneysel model çıktısıdır. Yatırım tavsiyesi değildir; gelecekteki fiyatı veya getiriyi garanti etmez.”** |
| M-03 / “Canlı Piyasa” yerine | **“Son veri fiyatı”**; yanında **“Kaynak: Yahoo Finance”** ve son bar zamanı görünür. |
| M-04 / “AI Güveni” yerine | **“Model uyumu (sezgisel)”** |
| M-05 / uyum skoru açıklaması | **“Model tahminlerinin birbirine yakınlığını özetler; doğruluk veya gerçekleşme olasılığı değildir.”** |
| M-06 / grafik lejandı | **“Şematik tahmin yolu”**, **“Model yayılımı (görsel)”**, **“Üst yayılım sınırı”** |
| M-07 / grafik açıklaması | **“Çizgi, son fiyat ile hedef tahmin arasındaki görsel geçiştir; ara adımlar için ayrı model tahmini değildir. Bant istatistiksel güven aralığı değildir.”** Sonraki Accepted ADR-0008 gözlem sunumu uyarlaması. |
| M-08 / veri penceresi | **“Kullanılan veri aralığı”**; yalnız eldeki DataFrame başlangıç/bitiş zamanı. Bu alan, modelin kesin eğitim penceresi diye adlandırılmaz. |
| M-09 / üç senaryo kartı başlığı | **“Üst yayılım senaryosu”**, **“Model ortalaması”**, **“Alt yayılım senaryosu”** |
| M-10 / sabit %30/%40/%30 olasılık metinleri yerine | **“Görsel senaryo — gerçekleşme olasılığı hesaplanmamıştır.”** |
| M-11 / model yayılımı geçersizse | **“Model yayılımı hesaplanamadı; senaryo ve bant gösterilemiyor.”** |

### A'nın davranış ve test sınırı

- Nitelik sınıflandırması mevcut `source` değerine dayanır: `yahoo_finance`
  sağlayıcı, `mock`/`demo`/`test-fixture` demo olarak gösterilir. Diğer değerler
  değiştirilmeden kaynak alanında görünür, türleri bilinmiyor kabul edilir.
- Cockpit analiz metadata'sı mevcut yanıtla birlikte saklanır; API alanları,
  veri modeli ve sağlayıcı politikasına yeni alan eklenmez.
- Eski analiz; yeni analiz isteği, analiz hatası veya sembol değişiminde
  temizlenir. Sayısal hesaplama veya finansal sinyal kuralları değişmez.
- RSI için başlangıç, sonuç, başarılı boş, atlanan sembol ve istek hatası
  durumu ayrılır; mevcut tekrar deneme düğmesi kullanılır.
- Legacy ML'de kaynak ve `data.index` zamanları kullanılır. Kayıtlı olmayan
  çekilme zamanı sonradan uydurulmaz. Model, confidence formülü ve bant hesabı
  bu metin/görünürlük diliminde değiştirilmez.
- Yayılım geçersizse senaryo/bant gösterimi durdurulur; mevcut ±%5 fallback
  sayıları gerçek model yayılımı gibi sunulmaz. Geçerli model tahmininin değeri
  değişmez. Sabit senaryo olasılıkları tamamen kaldırılır.
- Mock yardımcılar aktif akışa eklenmez. Mevcut gerçek RSS akışı “demo” diye
  etiketlenmez; duygu skoru ayrıca deneysel olarak açıklanır.
- A için yeni ADR gerekmiyor: ADR-0001/0003'ün mevcut görünürlük gereksinimi.
  Bu belge onayı, aşağıdaki B'nin veri sözleşmesini veya başka onay kapılarını
  kabul edilmiş saymaz.

### B — RSS/NLP sözleşmesi ve kalan model doğruluk işleri

| İş | Neden A ile tamamlanmış sayılamaz? | Sonraki karar |
|---|---|---|
| RSS kaynak yayın zamanı / elde etme zamanı | Eski `date`, geçersiz yayın tarihiyle elde etme zamanını ayırt etmiyordu. | [ADR-0004](../50-decisions/ADR-0004.md) Accepted ve uygulandı: yayın zamanı/durumu ile çekilme zamanı ayrılır; bilinmeyende sahte tarih üretilmez. |
| NLP hata durumu | Eski hata sonucu `0.0`, gerçek nötr skorla birleşiyordu. | [ADR-0004](../50-decisions/ADR-0004.md) Accepted ve uygulandı: başarısız skor `None` olur; hesaplanmış nötrden ve özetten ayrılır. |
| ML kesin eğitim penceresi / model değerlendirmesi | Ham DataFrame aralığı, temizlenmiş özellik/target eğitim aralığı değildir; uyum skoru doğruluk metriği değildir. | BL-001 ve AI evaluation işleri; bu onay model kalitesini veya kalibrasyonu doğrulamaz. |
| ML tahmin ufku / hedef tarihi | A aşamasında model gözlem hedefi ile kart/grafiğin iki farklı takvim başlangıcı uyumsuzdu. | Sonraki [ADR-0008](../50-decisions/ADR-0008.md) özel onayla uygulandı: h gözlem, gerçek son index, kesin hedef tarihi yok;0..h şematik eksen. A onayı tek başına bu kararı kapsamıyordu. |
| Sentiment etiket / filtre eşikleri | Etiket helper'ı ±0.2, özet sayacı/filtreler ±0.1 kullanır. | ADR-0004 mevcut eşikleri korur. Eşik uyumu ayrı hesaplama kararıdır; sessizce değiştirilmez. |

Bu sözleşme RSS sağlayıcısının yayın zamanının doğruluğunu bağımsız doğrulamaz;
yalnız eksik/belirsiz zamanı biliniyor gibi sunma hatasını giderir. ML model
doğruluğu ve sentiment eşik uyumu ayrı kapanış işleridir. Tarih/ufuk sunumu
sonraki Accepted ADR-0008 kapsamında düzeltildi; işlem takvimi hesaplanmaz.

## Kabul ve doğrulama

- Kontrollü API yanıtlarıyla Yahoo, açık demo, bilinmeyen kaynak ve eksik
  metadata senaryoları; değer/kaynak/zaman/uyarı aynı ekranda sınanır.
- Başarılı analizi izleyen hata ve sembol değişimi önceki sonucu kaldırır;
  sentetik fiyat fallback'i eklenmez.
- RSI başlangıç/boş/veri yok/hata durumlarının metinleri ve yeniden deneme
  gerçek Next.js ekranında koşulsuz doğrulanır; mevcut beş senaryo korunur.
- ML üst uyarısı tahmin çalıştırılmadan önce de görünür; model sonucu,
  uyum kartı ve grafik üzerinde deneysel nitelik korunur. Haber ekranı gerçek
  zaman garantisi vermez; RSS haber ile deneysel skor aynı şeymiş gibi sunulmaz.
- ML kartlarında sabit olasılık metni kalmaz; geçersiz yayılımda sentetik
  fallback senaryosu/model bandı yerine M-11 görünür.
- Uygun Streamlit kontrolleri kontrollü veriyle planlanır; yalnız kaynak kodu
  taraması tam legacy UI kanıtı sayılmaz. Yeni dış çağrı veya dependency gerekirse
  mevcut onay sınırları ayrıca değerlendirilir.
- Frontend build, deterministik UI, ilgili backend/legacy testleri ve CDSK
  belge kontrolü uygulanır; B açıkken BL-002/M1 tamamlandı denmez.

## Onay kaydı

- A ekran metinleri ve yukarıdaki mevcut-alan davranışları: **Onaylandı**.
- B RSS/NLP veri sözleşmeleri: **ADR-0004 Accepted**.
- Önceki genel “devam” talebi özel onay sayılmamıştı. Bu somut iki belge için
  sorulan onay sorusuna gelen yeni “devam et” yanıtı 2026-11-05 onayıdır.
- Uygulama/doğrulama kaydı: `docs/20-execution/2026-11-05-bl002-implementation.md`.

## İnceleme kaydı

Bağımsız belge/kod incelemesi sabit senaryo olasılıkları açığını belirledi;
M-09–11 ve davranış/kabul koşullarına eklendi. ML ufuk/tarih ve sentiment eşik
tutarsızlıkları kalan iş olarak açıkça kaydedildi. Çalışma zamanı testi yapılmadı.

Doğrulama: öneri/ADR bağlantıları ve Proposed durumları, 39 zorunlu CDSK
belgesi/11 Boot bölümü, 39 Python dosyasının `compile()` kontrolü ve değişen
dosyaların whitespace kontrolü geçti. Üretim kaynakları/dependency bildirimleri
değişmedi. Bu sonuç upstream CDSK validator değildir; belgelenen validator yolu
Windows ortamında yoktur. Önceki 60 backend / 5 UI testi bu belge adımında
yeniden çalıştırılmadı.
