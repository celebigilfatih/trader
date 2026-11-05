# Core Workflows

## Workflow 1 — Bir hisseyi teknik olarak inceleme

1. Kullanıcı teknik analiz ekranını açar.
2. Sembol, periyot, interval ve göstergeleri seçer.
3. Sistem Yahoo Finance üzerinden OHLCV verisini ister.
4. Teknik göstergeler, grafik, sinyal ve risk özeti üretilir.

### Success

Veri kaynağı ve analiz zamanı belli olan grafik ve açıklanabilir sonuç gösterilir.

### Failure and Recovery

Veri yoksa veya dış servis hata verirse analiz üretilmez; kullanıcı farklı sembol
veya zaman aralığı denemeye yönlendirilir.

## Workflow 2 — Hisse tarama

1. Kullanıcı tarama kuralı ve zaman aralığını seçer.
2. Sistem sembol kataloğunu sırayla tarar.
3. Koşulu sağlayan hisseler ölçümleriyle listelenir.

### Success

Sonuçlar kullanılan kural ve interval ile birlikte gösterilir.

### Failure and Recovery

Kısmi dış servis hataları başarısız sembolü atlayabilir. Gelecek geliştirmede
kaç sembolün başarıyla tarandığı ve kaçının hata verdiği ayrıca raporlanmalıdır.

## Workflow 3 — Haber ve duygu görünümü

1. Kullanıcı haber sayısı, duygu ve sembol filtresini seçer.
2. Sistem yapılandırılmış RSS kaynaklarını okur.
3. VADER/TextBlob skoru ve tespit edilen sembollerle haberleri gösterir.

### Success

Haber bağlantısı, kaynak, tarih ve hesaplanan duygu etiketi birlikte görünür.

### Failure and Recovery

Kaynak erişilemiyorsa hata gizlenmez; boş sonuç veya bağlantı uyarısı gösterilir.
