# Security Model

## Assets and Trust Boundaries

- Kod ve analiz algoritmaları.
- Yerel portföy/işlem örneği ve üretilebilen alarm/model dosyaları.
- Runtime'da verilebilen SMTP kimlik bilgileri.
- Güvenilmeyen sınırlar: Yahoo Finance yanıtları, RSS içerikleri ve haber
  bağlantıları.

## Threats and Controls

- Yanlış/eski finansal veri ve mock verinin gerçek sanılması: kaynak, zaman ve
  deneysel etiket zorunluluğu.
- Dış içerikten UI enjeksiyonu: haber metni HTML'e girmeden escape edilmeli;
  bağlantı şeması doğrulanmalıdır.
- Secret sızıntısı: `.env*` ve `.streamlit/secrets.toml` izlenmemelidir.
- Kaynak tüketimi: geniş taramalar ve model eğitimi sınırlandırılmalı; dış
  çağrılar timeout/retry bütçesi kullanmalıdır.

## Identity, Authorization and Audit

Kimlik, rol, yetki ve audit sistemi yoktur. Çok kullanıcılı veya internete açık
dağıtım öncesi bu alanlar ayrı güvenlik kararı gerektirir.

### Accepted Plan (MVP)

- Kapalı beta: davet + e-posta magic link ile doğrulanmış, oturum bazlı erişim.
- Tek kullanımlık token + kısa zaman penceresi + iptal edilebilir oturum modeli
  hedeflenir.
- Bildirim ve AI çıktıları, portföy ve analiz yetkileri ile karışmayacak şekilde
  model katmanı ayrılır.

## Secrets and Sensitive Data

Secret repository, log, ekran görüntüsü veya CDSK belgelerine yazılmaz. SMTP
parolası runtime secret store/environment üzerinden sağlanmalıdır.
`portfolio_data.json` finansal kullanıcı verisi içerebileceği için örnek/gerçek
veri niteliği ve saklama politikası ürün sahibiyle netleştirilmelidir.
