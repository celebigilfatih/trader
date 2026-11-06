# AI Architecture

## AI Use Cases

- Geçmiş fiyat/indikatör özelliklerinden ertesi gün fiyat/getiri tahmini.
- Haber metnini pozitif, nötr veya negatif olarak sınıflandırma.
- UI'daki “AI Tahminleri” ekranında model sonucu ve özellik önemini gösterme.

## Models, Tools and Data Boundaries

- Modeller yerel scikit-learn RandomForest/ensemble sınıflarıdır; uzak model API'si
  yoktur.
- NLP, VADER ve TextBlob ile yerelde yapılır.
- Girdi fiyat verisi Yahoo Finance'tan, haber verisi RSS kaynaklarından gelir.
- `SentimentAnalyzer` içindeki bazı skor, başlık, sosyal medya ve geçmiş
  fonksiyonları mock/demo veri üretir.
- 2026-11-05 kaynak/AST incelemesinde bu mock yardımcıların diğer Python
  modüllerinde doğrudan çağrısı bulunmadı. Aktif `show_news_feed`, gerçek RSS
  haber akışını kullanır; bu bulgu dinamik çağrıların veya tüm UI'nın çalışma
  zamanında doğrulandığı anlamına gelmez.

## Guardrails and Human Approval

- Tahmin finansal tavsiye veya otomatik işlem tetikleyicisi değildir.
- Mock ve deneysel çıktı görünür biçimde etiketlenmelidir.
- Model sonucu veri zamanı, eğitim penceresi ve temel hata metriği olmadan güvenilir
  kabul edilmemelidir.
- Emir verme ya da kullanıcı adına finansal aksiyon insan onayı olmadan eklenemez.

## Evaluation, Observability and Fallback

Model eğitim hataları sözlük içinde döndürülür; kapsamlı evaluation, drift,
registry veya üretim metriği yoktur. Veri ya da model başarısızsa teknik analiz
akışı çalışmaya devam edebilmelidir.

## Veri niteliği / deneysel çıktı sunumu

- Legacy ML üst uyarısı ve girişleri deneyseldir. Sezgisel skor “Model uyumu”
  olarak açıklanır; doğruluk veya gerçekleşme olasılığı değildir. Grafik ara
  adımları şematik geçiş, bandı model yayılımıdır; güven aralığı iddiası kaldırıldı.
- Sabit %30/%40/%30 senaryo olasılıkları kaldırıldı; geçersiz model yayılımında
  ±%5 fallback senaryosu/bandı gösterilmez. Geçerli model hesapları değişmedi.
- RSS bilinmeyen yayın zamanını bugünün saatiyle doldurmaz; NLP hatası veya
  sonlu olmayan sonuç `None/failed` olur. Hesaplanmış nötr ile ayrılır.
- Ekran metinleri `docs/00-product/BL002_DATA_DISCLOSURE.md` ve RSS/NLP
  sözleşmesi `ADR-0004` 2026-11-05 tarihinde kullanıcıca onaylandı.
- ADR-0008 gözlem ufku/takvim sunumu farkını kesin hedef tarihi kaldırarak kapattı.
  Sentiment etiket/filtre eşik farkı ve model kalibrasyonu ayrı açık işlerdir.

## ML hedef/ön işleme ve artifact — Accepted ADR-0007

Kullanıcı somut ADR-0007 sorusuna “Onaylıyorum.” dedi; sözleşme uygulandı.
Bilinmeyen hedef NaN, gerçek sıfır getiri0.0; kaynak satırında i+h<ilk test
konumu purge sınırıdır. Min50 etiket ve purge sonrası train/test>=2 gerekir.
Feature temizlik stateless; %1/%99 bounds ve iki scaler yalnız tutulan train'de
öğrenilir. Dört estimator fit/inference aynı scaled X ve inverse y kullanır.
Başarısız retrain eski advanced paketi kullanıma açık bırakmaz.

97 hedef ML testi, native dört-model smoke ve ayrı sabit label boundary ile
literal160→164 oracle'ı içerir. Her modelde feature sırası/ölçek zinciri ve gerçek
joblib temp roundtrip doğrulanır. Kaynak indicator'ların causal olduğu, piyasa
accuracy/kalibrasyon/kârlılığı veya genel M1 kapanışı iddia edilmez.

ML-01–06 bilinen dilimleri kapandı. Simple her eğitimde trained flag/cache'i
temizler; sonlu olmayan metric başarı sayılmaz. Importance RF'nin fitted kolon
metadata'sını kullanır; invalid inference error ve boş son tahmin cache'i bırakır.
Geçerli forest/signal/confidence/çıktı biçimleri aynı; yeni model veya artifact yok.
Ufuk cap/helper/gözlem sunumu da Accepted ADR-0008 ile aşağıda düzeltildi.
Process-wide warning susturma ve Windows uyarı
encoding'i bu dilimde değişmedi. Ayrıntı [BACKLOG](../20-execution/BACKLOG.md)
ve [execution kaydındadır](../20-execution/2026-11-06-adr0007-implementation.md).
Simple düzeltme kanıtı [ayrı kayıttadır](../20-execution/2026-11-06-simple-ml-fixes.md).

## ML-07/08 ve gözlem sunumu — Accepted ADR-0008 uygulandı

[ADR-0008](../50-decisions/ADR-0008.md) özel “onayladım, devam edelim” yanıtıyla
Accepted ve uygulandı. Tek get_prediction_return_bounds helper'ı h1±.10,
2–7±.25,8–30±.50,31+[-.70,1.00] verir; inverse sonrası bir kez clip edilir.
Fiyat sözlüğü için current_price bağlamı ve geçerli legacy return uyumu vardır;
mixed/invalid payload ValueError, sezgisel skor/strict eşikler aynı kalır.
Panel h gözlem/gerçek last_index/0..h şematik eksen ve aynı helper caption'ını
kullanır; gelecekteki kesin tarih üretilmez. Fiyat sözlüğü/format1/model parametreleri
aynı; UI'nın model-yayılımı sezgisi yeni helper'a bağlanmadı/kalibre edilmedi.
Kanıt [execution kaydında](../20-execution/2026-11-06-adr0008-implementation.md).

## Prompt and Model Versioning

Prompt tabanlı model yoktur; model hiperparametreleri kodda korunur. Yeni güvenilen
yerel joblib format_version=1, preprocessing.clip_bounds ve prediction_horizon
saklar. Sürümsüz/bozuk paket yeniden eğitim ister, load doğrulama bitmeden çalışan
state'i değiştirmez. Bu kontrol deserialization güvenliği veya sklearn sürümleri
arası uyumluluk değildir; destek matrisi/retention/registry ayrı kalır.
