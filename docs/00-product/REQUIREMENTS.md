# Requirements

| ID | Requirement | Source | Acceptance criteria | Status |
|---|---|---|---|---|
| REQ-001 | Kullanıcı desteklenen BIST sembolü için OHLCV verisini inceleyebilmelidir. | Mevcut README ve `data_fetcher.py` | Başarılı yanıtta OHLCV sütunları doğrulanır; boş yanıtta açıklanabilir hata gösterilir. | Observed |
| REQ-002 | Kullanıcı seçilen teknik göstergeleri ve sinyali görebilmelidir. | Mevcut uygulama | Göstergeler aynı veri setinden hesaplanır ve grafik/özet üzerinde görünür. | Observed |
| REQ-003 | Kullanıcı hisseleri teknik kurallarla tarayabilmelidir. | Mevcut uygulama | Sonuç, kuralı sağlayan sembol ve ölçümleri içerir. | Observed |
| REQ-004 | Haber sonuçları kaynak, tarih ve duygu etiketi içermelidir. | Mevcut uygulama | RSS verisi geldiyse dört alan da sonuç kartında gösterilir. | Observed |
| REQ-005 | Uygulama çıktıları yatırım tavsiyesi olarak sunulmamalıdır. | README | Risk uyarısı ilgili analiz akışlarında görünürdür. | Partially verified |
| REQ-006 | Mock ve deneysel sonuçlar gerçek veriyle karıştırılmamalıdır. | Kod inceleme riski | Kullanıcı arayüzünde kaynak/deneysel etiketleri görünürdür. | Proposed |
| REQ-007 | Çekirdek hesaplamalar dış ağ olmadan test edilebilmelidir. | CDSK kalite standardı | Deterministik fixture'larla birim test paketi geçer. | Proposed |
