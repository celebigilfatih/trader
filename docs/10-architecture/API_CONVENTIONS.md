# API Conventions

## Protocol and Versioning

Planlanan hedefte projede HTTP API olacak ve ana iş akışı `FastAPI` ile `/api/v1`
şeklinde sunulacaktır. MVP'nin başlangıçta tekdüze dağıtımı yerine, yeni API ve
next gen UI adım adım devreye alınır. Geçiş tamamlandığında versiyonlama ve hata
sözleşmesi ilk kez zorunlu hale gelir.

## Authentication and Authorization

Hedeflenen MVP’de kapalı beta için davet + passwordless e-posta link ile oturum
oluşturulur. API kimliği oturum bazlıdır ve roller/izinler sonraki iterasyonda
genişletilir.

## Errors, Idempotency and Pagination

Merkezî hata sözleşmesi `error_code`, `message`, `correlation_id` alanlarıyla
tanımlanır. Anahtar yazılan iş akışlarında idempotency ve job_id korunur.
Dış entegrasyonlar açık timeout/retry/backoff ve “veri eksikse nasıl degrade olur”
şeklinde sözleşmeye dayanır.

## Compatibility

Public fonksiyon imzaları, OHLCV sütunları ve çıktı sözlük anahtarları fiili iç
sözleşmelerdir. Geriye uyumsuz değişiklik test, çağıran analizi ve gerekiyorsa ADR
ile yapılmalıdır.
