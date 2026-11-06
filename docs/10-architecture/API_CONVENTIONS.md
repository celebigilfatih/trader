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

## Stochastic seçicisi — Accepted ADR-0009

`GET /api/v1/analysis/{symbol}?indicators=stoch` dış `stoch` anahtarı altında
son %K'yı float/null verir. Yalnız API projeksiyonu mevcut `stoch_k` serisine
eşler; hesaplayıcı/legacy K/D serileri değişmez. K=0 →0.0; warmup/flat NaN →null.
Karma istekte diğer seçiciler ve metadata korunur; D alanı/query seçicisi
eklenmez, `stoch_k/stoch_d` query adları400 kalır. Accepted
[ADR-0009](../50-decisions/ADR-0009.md) uygulaması OS-01 HTTP500 hatasını kapattı;
[gerçek HTTP regresyon kanıtı](../20-execution/2026-11-06-adr0009-implementation.md).

## VWEMA/Supertrend — Accepted ADR-0010 uygulandı

[ADR-0010](../50-decisions/ADR-0010.md) A paketi, mevcut vwema_5/vwema_20
scalar/null adlarını koruyarak matematiği geçmişe bağlı hacim ağırlığına geçirir.
Supertrend defaultta11 bardan önce fiyat/yön üretmez; API mevcut supertrend
alanında null verir. All-zero hacimde VWEMA da null'dır. Auth/provenance ve query
adları korunur; üç gerçek HTTP regresyonu eklendi, API paketi18/18 geçti.
[Uygulama kanıtı](../20-execution/2026-11-06-adr0010-implementation.md).

## Açık çoklu çıktı sınırı — OS-02

Gerçek analyzer ile fvg/order_block/bos/fvg_ob_combo/fvg_bos_combo/bollinger
bu skaler route'ta HTTP500 verir. Config'de bulunmak HTTP projeksiyonunun çalıştığı
anlamına gelmez. [ADR-0011](../50-decisions/ADR-0011.md) **Proposed A**, bu altı
seçiciyi açık400 ile reddetmeyi önerir; henüz kabul/uygulama yoktur. Structured
nesne/list endpoint'i bu önerinin sunduğu bir yetenek değildir.
