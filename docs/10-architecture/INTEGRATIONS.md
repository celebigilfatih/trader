# Integrations

| Integration | Purpose | Owner | Contract | Failure/fallback |
|---|---|---|---|---|
| Yahoo Finance / `yfinance` | OHLCV, anlık özet ve şirket bilgisi | `BISTDataFetcher` | Yahoo sembolü → normalize OHLCV veya `None` | Hata yazdırılır, `None`; merkezi timeout/retry yok |
| Investing TR RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Finans Gündem RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Bloomberg HT RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Anadolu Ajansı ekonomi RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| Hürriyet ekonomi RSS | Finans haberleri | `SentimentAnalyzer` | RSS entry → NewsItem | Kaynak atlanır, diğerleri sürer |
| SMTP | İsteğe bağlı e-posta alarmı | `AlertSystem` | Runtime SMTP config + alert | `False` döner; retry yok |
| Provider abstraction | Piyasa veri standardizasyonu | Hedef: `MarketDataProvider` adapterları | Sözleşmeli veri çıktısı (`source`, `as_of`, `staleness`) | Sağlayıcı değişiminde core logic etkilenmeden geçiş |

## Operational gaps

- Dış çağrılarda açık timeout ve ortak retry/backoff politikası yoktur.
- Veri lisansı, rate limit ve üretim kullanım koşulları doğrulanmamıştır.
- Sağlık metriği ve entegrasyon başına hata görünürlüğü yoktur.
- Dış çağrılar hedef mimaride timeout, retry, hata kodu ve fallback stratejisi ile
  standartlaştırılacaktır.
- Coolify iş takvimi ile günlük close-of-day snapshot akışları planlanmaktadır.
