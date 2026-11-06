# API Conventions

## Protocol and Versioning

Mevcut yerel FastAPI servisi `/api/v1` altında health, demo auth, sembol,
watchlist, analiz, RSI ve paper portföy akışlarını sunar. Kabul edilmiş API-first
geçiş adım adım sürer; üretim kalıcılığı/worker ve tüm legacy akışların taşınması
tamamlanmış değildir.

## Authentication and Authorization

Hedeflenen MVP’de kapalı beta için davet + passwordless e-posta link ile oturum
oluşturulur. API kimliği oturum bazlıdır ve roller/izinler sonraki iterasyonda
genişletilir.

## Errors, Idempotency and Pagination

Merkezî hata sözleşmesi `error_code`, `message`, `correlation_id` alanlarıyla
tanımlanır; mevcut HTTPException yanıtları bu alanları `detail` altında taşıyabilir.
Tüm hata yollarını normalleştiren middleware henüz yoktur. Yazılan iş akışlarında
idempotency/job_id ve pagination hedefleri tamamlanmış kabiliyet sayılmaz.
Dış entegrasyonlar açık timeout/retry/backoff ve “veri eksikse nasıl degrade olur”
şeklinde sözleşmeye dayanır.

## Compatibility

Public fonksiyon imzaları, OHLCV sütunları ve çıktı sözlük anahtarları fiili iç
sözleşmelerdir. Geriye uyumsuz değişiklik test, çağıran analizi ve gerekiyorsa ADR
ile yapılmalıdır.

OHLCV provider'a enjekte edilen fetcher, `get_stock_data(symbol, period=...,
interval=..., timeout_seconds=...)` keyword'ünü kabul etmelidir. Gerçek
`BISTDataFetcher` için yeni keyword isteğe bağlıdır; eski positional/legacy
çağrılar override vermeden aynı davranışı korur. Bu iç adaptör sözleşmesi
yeni HTTP endpoint veya response alanı eklemez.

## Stochastic seçicisi — Proposed sözleşme

OS-01: `stoch` seçimi gerçek hesaplayıcıda `stoch_k/stoch_d` üretirken route
`stoch` anahtarını arar ve HTTP500 verir. Mevcut yanıt seçilen ad başına
skaler/null biçimindedir; K/D anlamı tanımlanmamıştır.
[ADR-0009](../50-decisions/ADR-0009.md) Proposed: dış `stoch` anahtarını %K'ya
eşlemek, D'yi/yeni alanları API'ye eklememek. Özel karar onayı ve uygulaması
bekleniyor; bu bölüm mevcut çalışan API davranışı iddiası değildir.
