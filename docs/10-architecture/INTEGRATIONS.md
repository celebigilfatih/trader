# Integrations

| Integration | Purpose | Owner | Contract | Failure/fallback |
|---|---|---|---|---|
| Yahoo Finance / `yfinance` | OHLCV, anlık özet ve şirket bilgisi | `BISTDataFetcher` | Yahoo sembolü → normalize OHLCV veya `None` | `get_stock_data` isteğe bağlı history timeout kabul eder; hatada mevcut `None` sözleşmesi korunur. API provider sınırlı retry uygular. Diğer Yahoo metodları ortak bütçe altında değildir. |
| Investing TR RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Finans Gündem RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Bloomberg HT RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Anadolu Ajansı ekonomi RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Hürriyet ekonomi RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| SMTP | İsteğe bağlı e-posta alarmı | `AlertSystem` | Runtime SMTP config + alert | `False` döner; retry yok |
| Provider abstraction | Piyasa veri standardizasyonu | Mevcut: `MarketDataProvider` | `MarketDataResult` + `source`, `as_of`, `fetched_at`, `staleness_seconds` | 8 saniye history timeout, 2 ek deneme ve 1.5/3.0 saniye backoff; tükenmede `RuntimeError`, sentetik fiyat yok. |

## Mevcut timeout / hata davranışı

- Provider'ın mevcut `timeout_seconds` alanı `get_stock_data(...,
  timeout_seconds=...)` üzerinden gerçek `Ticker.history(timeout=...)` çağrısına
  iletilir. Değerler bu düzeltmeyle değiştirilmedi.
- Legacy çağrılar override vermezse yfinance'ın kendi history varsayılanı
  korunur. Enjekte fetcher aynı keyword'ü kabul etmelidir; repository çağıranları
  ve doubles test edildi. Bu parametre bir toplam işlem deadline'ı değildir.
- Kurulu yfinance 1.8.0, lazy timezone ediniminde kendi `timeout=10` değerini
  kullanır. Başka sürümler/SDK içindeki bütün HTTP yolları için ortak bütçe veya
  hard timeout doğrulanmış değildir; BL-007 destek matrisi ayrı kalır.
- RSS, Accepted [ADR-0005](../50-decisions/ADR-0005.md) uyarınca `requests`
  ile byte gövde alıp `feedparser` ile yerelde parse eder. Connect/read `(8, 8)`
  saniye; en fazla 2 ek deneme, 1.5/3.0 saniye bekleme uygulanır. Timeout/bağlantı
  ve HTTP 408/429/500/502/503/504 tekrar denenir. 429/503 + `Retry-After` varsa
  çağrı içinde tekrar ertelenir. Diğer HTTP hataları ve parser hataları retry edilmez.
  Redirect sonrası son HTTP sonucu denetlenir; hata gövdesi haber sayılmaz.
- Bozo parse kullanılabilir haberle `partial`, kullanılabilir haber olmadan
  `failed`; başarılı boş feed `empty` olur. Kaynaklar birbirinden yalıtılmıştır;
  NLP başarısızlığı RSS erişim tanısından ayrıdır. Sentetik haber/nötr skor yoktur.
- RSS ve Yahoo provider `last_fetch_diagnostics` listesinde kaynak, durum,
  adaptör deneme sayısı, monotonic toplam süre ve sınıflandırılmış hata kodu tutar.
  Liste her fetch'te yenilenir; özet `source_diagnostics` bağımsız kopyadır.
  Yahoo fetcher `None` verirse neden uydurulmadan `no_data` yazılır. Bu alanlar
  instance belleğindedir; API yanıtı veya health endpoint'i genişletilmedi.

## Operational gaps

- Bütün dış çağrıları kapsayan timeout/retry/backoff politikası yoktur;
  OHLCV ve RSS davranışları yukarıda açıklanmıştır; SMTP/diğer SDK yolları kapsam dışıdır.
- Veri lisansı, rate limit ve üretim kullanım koşulları doğrulanmamıştır.
- API process health endpoint'i vardır; bu dış kaynak sağlığı değildir.
  Yerel tanılar merkezi oran, alarm, kalıcı log veya SDK içi HTTP sayısı değildir.
- Dış çağrılar hedef mimaride timeout, retry, hata kodu ve fallback stratejisi ile
  standartlaştırılacaktır.
- Coolify iş takvimi ile günlük close-of-day snapshot akışları planlanmaktadır.
