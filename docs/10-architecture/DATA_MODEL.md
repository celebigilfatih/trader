# Data Model

## Canonical Entities

- **OHLCVSeries:** Tarih indeksli `Open`, `High`, `Low`, `Close`, `Volume`
  sütunlarından oluşan pandas DataFrame.
- **Symbol:** Yahoo Finance biçiminde kod (`THYAO.IS`) ve görünen şirket adı.
- **IndicatorSeries:** OHLCVSeries ile aynı zaman eksenindeki türetilmiş seri.
- **Signal/Alert:** Tür, mesaj, zaman ve hesaplanan değerden oluşan geçici çıktı.
- **NewsItem:** Başlık, açıklama, bağlantı, tarih, kaynak, duygu skoru ve tespit
  edilen semboller.
- **Portfolio snapshot:** `portfolio_data.json` içindeki holdings, transactions
  ve portfolio_history yapısı; aktif sahipliği kodda doğrulanamadı.
- **Cockpit entities (hedef):** `users`, `symbols`, `watchlists`,
  `analysis_snapshots`, `screener_runs`, `holdings`, `alert_rules`,
  `alerts`, `notifications`, `news_ingest`.

## Invariants and Ownership

- `BISTDataFetcher` dış veriyi OHLCV sözleşmesine normalize eder ve en az 15
  kayıt bekler.
- `TechnicalAnalyzer` gerekli beş OHLCV sütunu yoksa girdiyi reddeder.
- İndikatörler veri çekme katmanını değil yalnızca kendilerine verilen seriyi
  kullanır.
- Portföy dosyasının lifecycle sahibi mevcut kaynakta bulunmadığından bu veri
  için değişiklik yapılmamalıdır.
- Paper portföy hedef modeli yalnızca sembol, adet ve ortalama maliyet alanlarıyla
  başlar; kalan muhasebe metrikleri derived alanlardan hesaplanır.

## Lifecycle and Retention

Piyasa verisi, haberler ve çoğu analiz sonucu geçicidir. Streamlit cache/session
ömrü süreç ve kullanıcı oturumuyla sınırlıdır. Alarm CSV'si, model dosyaları ve
`portfolio_data.json` için retention politikası henüz tanımlı değildir.
- Hedef mimaride veriler PostgreSQL’de kalıcı tutulur; alarm/analiz çıktıları için
  TTL + retention planı uygulanır.

## Migration Policy

Şema migration sistemi yoktur. Kalıcı JSON/CSV/model formatını değiştiren iş,
önce sahiplik ve geri uyumluluk etkisini belirlemeli; gerekirse Proposed ADR ve
geri dönüş planı hazırlamalıdır.
