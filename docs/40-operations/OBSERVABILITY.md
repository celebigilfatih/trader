# Observability

## Logs, Metrics and Traces

Uygulama `print`, Streamlit hata/uyarı bileşenleri ve yer yer exception metni
kullanır. Yapılandırılmış log, merkezi metric veya distributed trace yoktur.

## Health and Alerts

Health endpoint ve otomatik alert yoktur. En kritik eksikler: veri kaynağı
başarı/hata/gecikme oranı, taranan sembol sayısı, cache yaşı ve model eğitim
sonucu.

## Dashboards and Ownership

Dashboard ve operasyon sahibi tanımlı değildir. Üretim kararıyla birlikte veri
sağlayıcı, UI hata oranı ve model/entegrasyon metriği sahipleri atanmalıdır.

## Sensitive Data Rules

Loglar SMTP parolası, kullanıcıya özgü portföy/işlem ayrıntısı veya gereksiz
haber içeriği taşımamalıdır. Hata için sembol, kaynak, zaman, sınıflandırılmış hata
türü ve correlation bilgisi yeterlidir.
