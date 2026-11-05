# Backup and Recovery

## Protected Data

- Kaynak kod ve CDSK belgeleri: Git repository.
- `portfolio_data.json`: finansal örnek/kullanıcı verisi olma ihtimali var;
  sahipliği açık değil.
- Runtime'da üretilebilen `alert_history.csv` ve model artifact'ları.
- Secret'lar backup kapsamına repository üzerinden girmez.

## Backup, Retention and Encryption

Resmi backup, retention ve encryption politikası yoktur. Üretim öncesi veri
sahibi, sınıflandırma, sıklık, saklama süresi ve şifreleme kararı zorunludur.

## Restore Procedure

Kod ve belgeler bilinen iyi Git commit'inden geri yüklenebilir. Yerel veri
artifact'ları için doğrulanmış restore prosedürü yoktur; dosyanın üzerine yazmadan
önce kopya alınmalı ve şema/JSON bütünlüğü kontrol edilmelidir.

## Restore Test Evidence

Henüz restore testi kaydedilmedi. Üretim hazırlığından önce örnek backup'tan
izole dizine restore ve uygulama okuma testi yapılmalıdır.
