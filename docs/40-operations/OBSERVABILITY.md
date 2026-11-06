# Observability

## Logs, Metrics and Traces

Uygulama `print`, Streamlit hata/uyarı bileşenleri ve yer yer exception metni
kullanır. Yapılandırılmış log, merkezi metric veya distributed trace yoktur.

## Health and Alerts

Yerel API'de `/api/v1/health` endpoint'i vardır; process/service erişimini
bildirir, Yahoo/RSS/SMTP veya hedef PostgreSQL/worker sağlığını doğrulamaz.
Otomatik kaynak-sağlık alarm hattı tamamlanmış değildir. En kritik eksikler:
veri kaynağı başarı/hata/gecikme oranı, taranan sembol sayısı, cache yaşı ve
model eğitim sonucu.

Accepted [ADR-0005](../50-decisions/ADR-0005.md) RSS ve Yahoo instance'larında
`last_fetch_diagnostics` sağlar: `source/status/attempts/elapsed_seconds/error_code`.
Süre monotonic, backoff dahil; deneme sayısı adaptör çağrısıdır. Her fetch listeyi
yeniler; haber özeti kendi kopyasını tutar. RSS erişimi kısmi/başarısızsa legacy
ekran onaylı eksik sonuç uyarısını ve kaynak durumlarını gösterir.
Bu tanılar süreç belleğindedir; endpoint, merkezi metric/log, alarm hattı, toplam
hard deadline veya SDK içi HTTP sayısı değildir. Fetcher `None` dönüşü Yahoo'da
`no_data` olarak kaydedilir; bilinmeyen kesinti nedeni tahmin edilmez.

## Dashboards and Ownership

Dashboard ve operasyon sahibi tanımlı değildir. Üretim kararıyla birlikte veri
sağlayıcı, UI hata oranı ve model/entegrasyon metriği sahipleri atanmalıdır.

## Sensitive Data Rules

Loglar SMTP parolası, kullanıcıya özgü portföy/işlem ayrıntısı veya gereksiz
haber içeriği taşımamalıdır. Hata için sembol, kaynak, zaman, sınıflandırılmış hata
türü ve correlation bilgisi yeterlidir.
