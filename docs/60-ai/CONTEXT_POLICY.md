# Context Policy

## Authority Order

Anayasa > kabul edilmiş ADR > ürün/kapsam > mimari > operasyon/kalite >
Project Boot > README/notlar > sohbet.

## Context Boundaries

- Koddan doğrulanabilen mevcut davranış “Observed” olarak kaydedilir; ürün hedefi
  veya bağlayıcı karar sayılmaz.
- Mock sentiment, deneysel ML ve canlı dış veri aynı kanıt düzeyinde ele alınmaz.
- Finansal iddia kaynak, zaman ve hesaplama yolu olmadan kalıcı bağlama eklenmez.
- `PROJECT_BOOT.md` özet, ayrıntılı docs ve kod kanıttır; özet ayrıntılı kaynağın
  yerine geçmez.

## Sensitive Data

Secret, SMTP parolası, kişisel veri ve gerçek kullanıcı portföy/işlem ayrıntısı
prompt veya kalıcı belgeye kopyalanmaz. Örnek veri gerekiyorsa anonim, sentetik ve
açıkça etiketli olmalıdır.
