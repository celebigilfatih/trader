# Runbook

## ADR-0010 hesaplama sürümü geçişi

Accepted VWEMA/Supertrend düzeltmesinden önce üretilen hesap cache'leri yeni
sonuçlarla karıştırılmaz; bu göstergeleri kullanan süreç/cache normal sürüm
geçişinde yenilenir. Eski bu özelliklerle eğitilen optional model yeniden eğitim
ister; format1 indikatör semantiğini otomatik tespit etmez. Bu uygulama çalışması
kullanıcı servisini yeniden başlatmadı veya artifact'ını açmadı/migrate etmedi.
Rollback helper/kart/test dilimini birlikte geri alır; VST-01–03 tekrar açılır.
[Karar](../50-decisions/ADR-0010.md) ve
[kanıt](../20-execution/2026-11-06-adr0010-implementation.md).

## Prerequisites

- Python 3 ve `venv`/`pip`.
- `requirements.txt` bağımlılıklarını kurmak için internet.
- Piyasa/haber ekranları için Yahoo Finance ve RSS kaynaklarına ağ erişimi.

## Setup and Start

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run.py
```

Alternatif başlangıç: `streamlit run app.py`. Beklenen yerel adres
`http://localhost:8501`.

Geçiş tamamlandığında:
- `uvicorn app.main:app --reload` (API)
- `npm run dev` (Next.js UI, hedef konfigürasyona göre)
- Coolify cron: BIST kapanışından sonra `18:20` planlı snapshot/scan çalıştırması.

## Stop and Recovery

- Durdurmak için terminalde `Ctrl+C`.
- UI state bozulursa Streamlit sürecini yeniden başlat.
- Veri alınamıyorsa önce farklı sembol/periyot dene, sonra Yahoo/RSS ağ
  erişimini kontrol et.
- Dependency sorunu varsa aktif Python/venv'i doğrula ve requirements kurulumunu
  tekrar çalıştır.

## Common Incidents

- **“Veri alınamadı”:** ağ, sembol, interval ve sağlayıcı erişimini kontrol et;
  sahte veriyle devam etme.
- **Haber listesi boş:** tek kaynak yerine diğer RSS sonuçlarını kontrol et;
  kaynak bazlı hata görünürlüğü şu an sınırlıdır.
- **Yavaş tarama:** tarama çok sayıda seri dış çağrı yapar; işlemi iptal edip daha
  dar akış kullan. Kalıcı çözüm BL-003 kapsamındadır.
- **Import error:** doğru venv'i aktive et ve `requirements.txt` kurulumunu
  doğrula.
- **Bildirim kuyruğu dolduysa:** outbox durumu sıfırlanmadan önce başarısız kayıtları
  ve tekrar deneme sayısını kontrol et.
- **Port 8501 kullanımda:** mevcut Streamlit sürecini güvenli biçimde kapat veya
  manuel Streamlit komutunda farklı port seç.

## Yerel ML modeli kurtarma — ADR-0007

- Sürümsüz/eski joblib paketini yeni kod otomatik migrate etmez; model yeniden
  eğitilir. Kullanıcı dosyası silinmez veya üstüne otomatik yazılmaz.
- Yeni eğitim başarısızsa eski model tahmine açık kalmaz; veri/ufuk ve en az50
  geçerli etiket sınırı kontrol edilir (eksiksiz seride en az50+h ham gözlem).
- Save/load yalnız güvenilen yerel artifact içindir. Bozuk paket yükleme çalışan
  instance'ı değiştirmez; format kontrolü joblib deserialization güvenliği sağlamaz.
  Desteklenen sürümler matrisi ayrı açık iştir.

## Escalation and Ownership

Kod sahibi/operasyon sorumlusu repository'de tanımlı değildir (`TBD`). Finansal
doğruluk, veri lisansı, güvenlik veya üretim olayı ürün sahibi ve atanacak teknik
sorumluya yükseltilmelidir.
