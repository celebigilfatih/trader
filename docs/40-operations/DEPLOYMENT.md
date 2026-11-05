# Deployment

## Environments

- **Local development/prototype:** Belgelenen tek ortam.
- **Test/staging/production:** Hostinger VPS + Coolify olarak hedeflenir.

## Release Steps

Hedef dağıtım yaklaşımı: Coolify’de API container + web container + job worker
container'larıyla yönetilen dağıtım. PostgreSQL ayrı servis olarak kullanılır.
Release adımı artık en az: dependency kilidi, env/secrets yönetimi, health check,
otomatik yedekleme, cron job doğrulama ve rollback denemesini içerir.

## Verification

Yerel/QA/prod adımında ana akışların çalışması, planlı close-of-day snapshot,
alarm kuyruğu, mock/provenance etiketi ve sağlık metrikleri doğrulanır.

## Rollback

Yerel değişikliklerde önceki Git commit'ine dönmek mümkündür; veri dosyaları
ayrıca korunmalıdır. Üretim tarafında Coolify rollback ve PostgreSQL yedek geri
yükleme hattı hedeflenir.
