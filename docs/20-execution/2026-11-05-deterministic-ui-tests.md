# Deterministik cockpit tarayıcı doğrulaması

**Kapsam:** BACKLOG sıralı iş 3; gerçek Next.js ekranında Playwright route
yanıtlarıyla test. Üretim API/UI koduna mock modu eklenmez. Mevcut canlı smoke
ayrı kalır; backend veya Yahoo/RSS/SMTP bu deterministik akışta kullanılmaz.

## Plan ve kabul

- [x] `scripts/run_frontend_ui_smoke.mjs` içine `--deterministic` ve isteğe
  bağlı `--browser-channel` ekle; deterministik modda eksik Playwright başarısız
  çıkış üretir. Canlı moddaki yenileme sonrası yanıt bekleyicisini tıklamadan
  önce kur.
- [x] `scripts/frontend_ui_scenarios.mjs` içinde bağımsız browser context'leri
  ve sabit test verisi kullan. Beklenmeyen API ve dış ağ istekleri koşuyu
  başarısız kılar; service worker engellenir. Frontend kaynakları yerelden gelir.
- [x] Başarı: demo auth, watchlist, analiz, paper ekleme/silme, RSI satırındaki
  fiyat/gösterge/kaynak/zaman/yaşlanma görünür; logout oturumu temizler.
- [x] Boş sonuç ve tüm sembollerde veri yokluğu eski başarılı satırları temizler.
- [x] Kısmi veri ve iki uyarı kodu koşulsuz görünür; göster/gizle tercihlerinin
  her ikisi yenileme ve yeniden tarama sonrası korunur. Yeni context temiz başlar.
- [x] HTTP hata yanıtı eski satır/uyarıları temizler; yeniden deneme başarıyla
  sonuçlanır. Bütün yanıt bekleyicileri aksiyondan önce kurulur.
- [x] Node sözdizimi, gerçek tarayıcı koşusu, frontend üretim derlemesi ve belge
  tutarlılığı kontrol edilir. Sonuçlar README, TEST_STRATEGY, BACKLOG,
  CHANGELOG ve PROJECT_BOOT handoff'a yazılır; M1 bütünü kapatılmaz.

## Karar sınırları

Yeni ürün metni, mimari, şema, servis veya bağımlılık kararı alınmaz. Lockfile
sürümleriyle yerel frontend kurulumu yapılır; test fixture'ları gerçek veri
olarak yayınlanmaz. Test başlangıcı çalışan yerel frontend gerektirir; yeni
Makefile hedefi mevcut canlı smoke hedefinin yanında açıkça ayrılır.

## Doğrulama kaydı

- Başlangıç: `codex/m1-pattern-signal-tests`, HEAD `6ed1a59`; önceki patern/sinyal
  testleri ve belge değişiklikleri korunuyor. Yeni commit/push yapılmaz.
- `npm ci --ignore-scripts --no-audit --no-fund` eşdeğeriyle lockfile kurulumu
  yapıldı; npm yerelde bulunmadığı için bundled pnpm üzerinden geçici npm CLI
  çalıştırıldı. Bildirim/lockfile değişmedi; Next.js kurulum güvenlik uyarısı açık.
- Playwright 1.62.1 ve kurulu headless Edge ile yerel 3030 portunda `next dev`
  ve üretim `next start` ekranlarında 5/5 senaryo geçti. Son üretim koşusu,
  yeniden denemede hata metninin kaybolması/başarı metninin dönmesini de sınadı.
- Tercih kaydı sadece test tarayıcısında devre dışı bırakıldı; üçüncü senaryo
  beklenen biçimde başarısız oldu. Playwright arama yollarında yokken runner
  exit 1 verdi. Üretim dosyaları bu kontroller için değiştirilmedi.
- Next.js build, iki Node scriptinin sözdizimi, 39 Python dosyası sözdizimi,
  manifestteki 39 zorunlu belge/11 Boot bölümü ve whitespace kontrolü geçti.
  Bu yapısal kontrol upstream CDSK validator değildir; validator ortamda yok.
- Bağımsız inceleme engelleyici bulgu üretmedi. Analiz fixture'ı UI'nın tükettiği
  alanlarla sınırlıdır; tüm analiz API sözleşmesi backend testlerinin konusudur.
  Genel hata/boş sonuç metni değiştirilmedi; BL-002 bağlamında kaydedildi.
- Canlı UI smoke ve backend testleri bu frontend adımında yeniden koşulmadı.
  Yeni Makefile hedefi mevcut CI'ye otomatik bağlanmadı; M1/M2 açık.
