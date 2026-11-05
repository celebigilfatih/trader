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

## Prompt and Model Versioning

Prompt tabanlı model yoktur. Model parametreleri kodda sürümlenir; kaydedilen
joblib dosyaları için sürüm/uyumluluk standardı henüz yoktur.
