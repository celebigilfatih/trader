# Coding Standards

## Language and Framework Rules

- Python kaynakları UTF-8, dört boşluk girinti ve anlamlı type hint/docstring
  kullanır.
- Streamlit widget key'leri kararlı ve benzersiz olmalıdır.
- Finansal hesaplamalar UI HTML/CSS üretiminden ayrılmalı; yeni domain mantığı
  `modules/` altında test edilebilir saf fonksiyon/sınıflara eklenmelidir.
- Geniş `except:` ve sessiz hata yutma yeni kodda kullanılmaz; beklenen hata
  türü ve kullanıcı/operasyon etkisi belirtilir.

## Repository Conventions

- `app.py` composition root/UI, `modules/` domain ve entegrasyon kodudur.
- Sembol ve indikatör sabitleri `modules/config.py` içinde merkezi tutulur.
- Secret ve kullanıcıya özel veri kaynak koda yazılmaz.
- Yeni kalıcı artifact'ın sahibi, formatı, retention ve ignore/backup davranışı
  belgelenir.

## Error, Security and Logging Rules

- Dış çağrılar açık timeout, sınırlı retry/backoff ve anlaşılır fallback
  kullanır.
- Gerçek, gecikmeli, cache, mock ve tahmin verisi birbirinden görünür biçimde
  ayrılır.
- Finansal sinyal logları secret/kişisel veri içermez; hata bağlamı sembol,
  kaynak ve zamanla izlenebilir olur.
- Kullanıcıya gösterilen dış metin HTML'e girmeden escape edilir; URL şeması
  güvenli listeyle doğrulanır.

## Formatting and Static Analysis

- **Format:** Araç yapılandırılmamış (`TBD`; öneri onay sonrası Black/Ruff).
- **Lint:** Yapılandırılmamış (`TBD`).
- **Typecheck:** Yapılandırılmamış (`TBD`).
- **Mevcut minimum:** Tüm `.py` dosyaları derleyiciyle sözdizimi kontrolünden
  geçmelidir.
