# Requirements

| ID | Requirement | Source | Acceptance criteria | Status |
|---|---|---|---|---|
| REQ-001 | Kullanıcı desteklenen BIST sembolü için OHLCV verisini inceleyebilmelidir. | Mevcut README ve `data_fetcher.py` | Başarılı yanıtta OHLCV sütunları doğrulanır; boş yanıtta açıklanabilir hata gösterilir. | Observed |
| REQ-002 | Kullanıcı seçilen teknik göstergeleri ve sinyali görebilmelidir. | Mevcut uygulama | Göstergeler aynı veri setinden hesaplanır ve grafik/özet üzerinde görünür. | Observed |
| REQ-003 | Kullanıcı hisseleri teknik kurallarla tarayabilmelidir. | Mevcut uygulama | Sonuç, kuralı sağlayan sembol ve ölçümleri içerir. | Observed |
| REQ-004 | Haber sonuçlarında kaynak, yayın zamanı durumu ve deneysel duygu hesabının durumu açık olmalıdır. | Mevcut uygulama + Accepted ADR-0004 | Geçerli yayın zamanı gösterilir; eksik/geçersiz zamanda tarih uydurulmaz ve çekilme zamanı ayrılır. Başarısız duygu hesabında skor/etiket null kalır; hesaplanmış sıfır nötrden ayrılır. | Onaylı RSS/legacy dilimi verified; genel kapsam Partially verified |
| REQ-005 | Uygulama çıktıları yatırım tavsiyesi olarak sunulmamalıdır. | README | Risk uyarısı ilgili analiz akışlarında görünürdür. | Partially verified |
| REQ-006 | Mock ve deneysel sonuçlar gerçek veriyle karıştırılmamalıdır. | Kod inceleme riski + onaylı BL-002 metinleri | Kullanıcı arayüzünde kaynak/deneysel etiketleri görünürdür. | Partially verified |
| REQ-007 | Çekirdek hesaplamalar dış ağ olmadan test edilebilmelidir. | CDSK kalite standardı | Deterministik fixture'larla birim test paketi geçer. | Partially verified |

2026-11-06 kanıtı ve kalan aktif yol eksikleri [M1 kapanış kontrolündedir](../20-execution/M1_CLOSEOUT.md).
366 backend testi dış socket engelli geçti; bu sonuç bütün hesaplama ve legacy
tarama yollarının doğrulandığı anlamına gelmez. Cockpit 10 deterministik browser
senaryosu önceki checkpoint kanıtıdır; güncel pakette 17 kontrollü legacy AppTest
vardır. REQ-004'ün önceki zorunlu tarih/skor metni Accepted
[ADR-0004](../50-decisions/ADR-0004.md) ile çeliştiği için yukarıda uyumlandı;
yeni veri sözleşmesi veya ürün kararı alınmadı.

Son momentum dilimi Stochastic/Williams %R/CCI çekirdeğine 17 test ekledi;
OS-01 Stochastic API anahtar eşleşmesi açık kaldı. REQ-002/007'nin genel
tamamlandığı iddia edilmez; ayrıntı kapanış matrisinde izlenir.
