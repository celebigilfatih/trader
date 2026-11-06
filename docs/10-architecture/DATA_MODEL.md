# Data Model

## Canonical Entities

- **OHLCVSeries:** Tarih indeksli `Open`, `High`, `Low`, `Close`, `Volume`
  sütunlarından oluşan pandas DataFrame.
- **Symbol:** Yahoo Finance biçiminde kod (`THYAO.IS`) ve görünen şirket adı.
- **IndicatorSeries:** OHLCVSeries ile aynı zaman eksenindeki türetilmiş seri.
- **Signal/Alert:** Tür, mesaj, zaman ve hesaplanan değerden oluşan geçici çıktı.
- **NewsItem:** Başlık, açıklama, bağlantı, kaynak ve tespit edilen semboller;
  `published_at` (offset'li ISO veya `None`), `published_time_status`
  (`valid/missing/invalid`), UTC `fetched_at`; `date` yalnız bilinen yayın
  zamanının gösterimidir, bilinmeyende boş string. `sentiment_status`
  (`computed/failed`), nullable `sentiment_score` ve `sentiment_label`.
- **Portfolio snapshot:** `portfolio_data.json` içindeki holdings, transactions
  ve portfolio_history yapısı; aktif sahipliği kodda doğrulanamadı.
- **Cockpit entities (hedef):** `users`, `symbols`, `watchlists`,
  `analysis_snapshots`, `screener_runs`, `holdings`, `alert_rules`,
  `alerts`, `notifications`, `news_ingest`.

## Invariants and Ownership

VWEMA/Supertrend için [ADR-0010](../50-decisions/ADR-0010.md) **Accepted ve uygulandı**:
ilk pozitif hacimden başlayan VWEMA; Supertrend fiyat/yön çiftinin aynı
hazır olma sınırı; NaN/invalid/empty ve eksik kart sözleşmesi. Konfigürasyon ve
ilgili fiyat/hacim validasyonu kısa/boş dönüşten önce; hata atomik ValueError'dur.
Yeni alan/kalıcı migration/artifact formatı yoktur. Bu özelliklerle eski
matematikte eğitilen optional modeller yeniden eğitim ister; format1 bunu otomatik
tespit etmez. Kullanıcı artifact'ı açılmadı/migrate edilmedi.

- `BISTDataFetcher` dış veriyi OHLCV sözleşmesine normalize eder ve en az 15
  kayıt bekler.
- `TechnicalAnalyzer` gerekli beş OHLCV sütunu yoksa girdiyi reddeder.
- İndikatörler veri çekme katmanını değil yalnızca kendilerine verilen seriyi
  kullanır.
- Risk/pozisyon okuyucularında geçerli RSI/MACD `0` eksik sayılmaz; None/NaN
  katkı oluşturmaz. DayTrader kayıt yönü add/close'da BUY/SELL olmalıdır;
  reddedilen yön, mevcut aktif kayıt/geçmişi değiştirmez. Kayıt alanları aynı.
- DayTrader private ATR hesabı istenen period kadar veri ister; entry için
  fiyat ve ATR pozitif/sonlu olmalıdır. Geçersiz girdi ValueError verir;
  sentetik HOLD/fiyat veya NaN seviyeli geçerli sonuç üretilmez.
- RSS bilinmeyen yayın zamanını çekilme zamanı ile doldurmaz; haberler bilinen
  zamanların gerçek anına göre sıralanır, bilinmeyenler en sondadır.
- NLP hatası/sonlu olmayan skor `None/failed` olur; gerçek hesaplanmış `0.0`
  nötr kalır. Özet/filtre yalnız hesaplanmış skorları kullanır; skorsuz haberi
  gerçek haber listesinden düşürmez. Skorsuz özette ortalama/etiket `None`.
- Haber özeti `scored_news` / `failed_sentiment_count` taşır; `total_news`
  bütün alınan gerçek haberleri sayar. `last_updated` özet oluşturma zamanıdır.
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

2026-11-05 tarihinde [ADR-0004](../50-decisions/ADR-0004.md) kabul edildi ve
RSS/özet sözleşmesi ile legacy okuyucular birlikte güncellendi. Bellek içi
skorun nullable olması bilinçli sözleşme değişikliğidir; kalıcı model/migration
veya kullanıcı veri dosyası değişmedi. Model/altyapı geçişleri ayrı onay ister.

Şema migration sistemi yoktur. Kalıcı JSON/CSV/model formatını değiştiren iş,
önce sahiplik ve geri uyumluluk etkisini belirlemeli; gerekirse Proposed ADR ve
geri dönüş planı hazırlamalıdır.

[ADR-0005](../50-decisions/ADR-0005.md) Accepted ve uygulandı. RSS/Yahoo instance
`last_fetch_diagnostics` listesi her çağrıda yenilenir. Kayıt alanları `source`,
`status` (`success/partial/empty/failed`), `attempts`, negatif olmayan monotonic
`elapsed_seconds`, nullable `error_code`'dur. Haber özeti aynı fetch'in bağımsız
`source_diagnostics` kopyasını taşır; sonraki fetch eski özeti değiştirmez.
Tanı RSS fetch/parse içindir; NLP skoru ayrı kalır. Ham exception/haber/kullanıcı
verisi taşımaz. API response alanları veya kalıcı şema değişmedi.

Accepted [ADR-0006](../50-decisions/ADR-0006.md) uygulandı. OTT üreticisi aynı
index'li `ott` fiyat, `ott_signal` buy/sell ve float `ott_trend` ±1 Series'lerini
verir. İlk gözlemde fiyat VAR'a seed edilir; sinyal/yön eksiktir, ilk yön
crossover değildir. Boş girdide üç boş Series; tek gözlemde yalnız fiyat vardır.
Close pozitif/sonlu reel fiyat olmalıdır; eksik/nonfinite/pozitif olmayan veya
fiyat olmayan tür ValueError verir ve önceki hesap/girdi değişmez. Kullanıcı
DataFrame'i korunur. VAR/config/recurrence/ağırlıklar aynı; yeni alan bellektedir,
kalıcı model/migration veya HTTP API alanı eklenmedi.

### Accepted ML sözleşmesi — ADR-0007 uygulandı

[ADR-0007](../50-decisions/ADR-0007.md) kullanıcı “Onaylıyorum.” yanıtıyla Accepted
ve uygulandı. Hedef kuyruğu/hesaplanamayan hedef NaN; gerçek sıfır getiri 0.0.
Train/test kaynak index'i artan/benzersiz; i+h<ilk test kaynak konumu purge sınırı.
En az50 geçerli etiket, purge sonrası train/test en az2; h gözlem sayısıdır.
Feature temizlik stateless; %1/%99 bounds ve iki scaler yalnız tutulan train'de
öğrenilir, bütün estimator'lar aynı dönüşüm ve target inverse kullanır.

Joblib mevcut beş alana format_version=1, preprocessing.clip_bounds ve pozitif
tam prediction_horizon ekler. Eksik/bozuk/sürümsüz paket ValueError ile yüklemeden
önce doğrulanır; çalışan instance korunur. Legacy artifact yeniden eğitim ister;
hiçbir kullanıcı dosyası açılmadı/migrate edilmedi. Yeni train denemesi eski fitted
state'i iptal eder; başarıda yalnız metric aşamasını da geçen modeller yayımlanır.
Güvenilen joblib kontrolü deserialization güvenlik garantisi veya sürümler arası
uyumluluk değildir. Kanıt [execution kaydındadır](../20-execution/2026-11-06-adr0007-implementation.md).

### Accepted ML helper/ufuk sunumu — ADR-0008

[ADR-0008](../50-decisions/ADR-0008.md) uygulandı: float price sözlüğü aynı,
confidence helper optional current_price ile price/current-1 kullanır; legacy
nested return yolunu korur. Mixed/invalid/error payload ValueError, skor alanları
aynı; ensemble skoru üretilmez. Tek aralık cap helper'ı model ve caption'ı besler.
UI h gözlem/gerçek son index gösterir, bilinmeyen timestamp üretmez; şematik0..h
ekseninde0 actual anchor, h hedef, yayılım1..h. Format1/DB/HTTP alanları aynı.
