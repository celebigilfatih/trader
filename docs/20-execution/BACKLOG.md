# Backlog

## Bekleme ve kaldığımız yer — 2026-11-05

Kullanıcı isteğiyle uygulama geliştirmesine şimdilik ara verildi. Bu kayıt yeni
özellik uygulama, commit veya push yetkisi değildir. Yeniden devam istendiğinde
önce AGENTS.md okuma sırası ve güncel Git durumu kontrol edilir.

- **CDSK:** 0.1.0 belge/çalışma iskeleti kurulu; M0 tamamlandı olarak kayıtlı.
  Yapısal validator'ın geçmesi, tüm ürün veya operasyon gereksinimlerinin
  tamamlandığı anlamına gelmez. Constitution hâlâ Draft ve TBD alanları var.
- **M1:** Kısmi; son kod doğrulamasında 35 backend testi geçti. Aşağıdaki
  sıralı beş işin ilk dar adımı tamamlandı, kalanları açık.
- **M2:** Yerel demo auth, watchlist ve paper portföy temeli var; gerçek davet
  operasyonu, alarm/bildirim ve günlük snapshot kapanışı açık.
- **M3:** Üretim hazırlığı Proposed; PostgreSQL kalıcılığı, worker, deployment,
  backup/restore, rollback ve operasyon sahipliği tamamlanmadı.
- **Git:** Yerel dal `main`; `origin` fetch/push adresi
  `https://github.com/celebigilfatih/trader.git`. Bu kontrol yalnızca yerel
  remote ayarını doğrular. Önceki bekleme kontrolünde CDSK belgeleri,
  `backend/`, `frontend/` ve başka yeni dosyalar untracked idi. Kullanıcı
  sonrasında açıkça commit + push istedi; bu teslim bunları sürümlemek içindir.
  Push öncesi uzak `main`, yerel başlangıç commit'i `2ca5644` ile eşleşti;
  push başarısı ayrıca uzak ref üzerinden doğrulanmalıdır.

| ID | Outcome | Priority | Dependencies | Acceptance criteria | Status |
|---|---|---|---|---|---|
| BL-001 | Deterministik çekirdek testleri | P0 | Test fixture'ları | Teknik analiz, patern, sinyal ve ML yardımcıları ağ olmadan test edilir. | In Progress |
| BL-002 | Gerçek/mock veri ayrımı | P0 | Ürün metni onayı | Mock sentiment ve deneysel ML her ilgili ekranda görünür etiketlenir. | In Progress |
| BL-003 | Dış entegrasyon dayanıklılığı | P0 | Timeout/retry kararı | Yahoo/RSS çağrılarında timeout, sınırlı retry/backoff ve hata metriği bulunur. | In Progress |
| BL-004 | README ve lisans doğruluğu | P1 | Ürün sahibi | Placeholder URL/iletişim temizlenir; lisans iddiası gerçek LICENSE ile uyumlu olur. | Backlog |
| BL-005 | Repository hijyeni | P1 | Veri sahipliği kararı | İzlenen `__pycache__` dosyaları ve orphan veri/artifact'lar güvenli biçimde ele alınır. | Backlog |
| BL-006 | `app.py` modülerleştirme planı | P1 | Mimari ADR gerekebilir | UI sayfaları ve orchestration küçük, test edilebilir modüllere ayrılacak planla tanımlanır. | Proposed |
| BL-007 | Bağımlılık ve Python sürümü tabanı | P1 | Destek matrisi kararı | Desteklenen Python sürümü ve yeniden üretilebilir dependency politikası doğrulanır. | Proposed |
| BL-008 | Veri/lisans ve dağıtım kararı | P0 before production | Ürün sahibi + hukuk/operasyon | Sağlayıcı koşulları, hosting, privacy ve rollback Accepted ADR'larla kayıtlıdır. | Accepted |
| BL-009 | Deploy hedefi ve operasyon hattı kurulumu | P0 | ADR-0002 | Coolify cron, PG backup, health ve rollback ile günlük snapshot iş hattı kurulur. | In Progress |

## Doğrulanmış durum — 2026-11-05

- BL-001 kısmen tamamlandı: provider/servis/API testlerine ek olarak gerçek
  `TechnicalAnalyzer` için 12 ağsız test ve gerçek hesaplayıcılı 3 API testi var.
  Toplam 35 backend testi geçiyor. Patern, sinyal ve ML yardımcılarının tamamı
  kapsanmadığı için BL-001 ve M1 açık kalır.
  Bağımsız inceleme ek açığı: MACD/Bollinger sabit seri testleri geçiyor;
  dönem/işaret/standart sapma çarpanı hatalarını yakalayacak bağımsız beklenen
  değerli değişken seri fixture'ları henüz eklenmedi (BL-001 içinde açık).
- BL-008 satırındaki `Accepted`, yalnızca ADR-0002'deki mimari hedefi ifade eder;
  veri lisansı, privacy ve operasyon onaylarının tamamlandığı anlamına gelmez.
- BL-009 hedefi kabul edilmiştir; mevcut yerel API'nin çalışması PostgreSQL,
  snapshot worker, backup veya üretim dağıtımının tamamlandığına kanıt değildir.

## Sıradaki beş iş (mevcut kapsam, yeni ürün kararı değil)

| Sıra | İş / dosyalar | Kabul ve doğrulama | Durum / kapı |
|---|---|---|---|
| 1 | Gerçek hesaplayıcı regresyonları: `backend/tests/test_technical_analysis.py`, `backend/tests/test_api_contracts.py` | EMA/SMA/VWMA, RSI yön/karma seri/21–22 bar sınırı, MACD/Bollinger, destek-direnç, giriş verisinin korunması; API'de gerçek hesaplama ve kısa veri uyarısı ağsız geçer. `make test-backend` | Bu dar adım tamamlandı; BL-001 bütünü değil |
| 2 | Patern ve sinyal testleri: `modules/pattern_recognition*.py`, `modules/alert_system.py`, yeni backend testleri | Sentetik tepe/dip ve eşik-altı/üstü girdilerle olumlu/olumsuz sonuçlar bağımsız beklenen değerlerle doğrulanır; ağ/SMTP çağrısı yapılmaz. Önce mevcut davranış okunur; doğrulanmış hata ayrı testle üretilir. | Sıradaki uygulama adımı; finansal davranış değişikliği gerekiyorsa ayrıca değerlendirilir |
| 3 | Deterministik tarayıcı senaryoları: `scripts/run_frontend_ui_smoke.mjs` | Teste özel kontrollü API yanıtlarıyla başarı/boş sonuç/veri yokluğu ve RSI tercih kalıcılığı koşulsuz sınanır; yanıt bekleyicileri tıklamadan önce kurulur; canlı smoke ayrı raporlanır. | Bekliyor; üretim koduna mock modu eklenmez |
| 4 | Veri niteliği görünürlüğü: `frontend/app/page.js`, legacy sentiment/ML ekranları | Her ilgili ekranda gerçek/mock/deneysel ayrımı ile kaynak ve zaman görünür; hatada sentetik fiyat gerçekmiş gibi sunulmaz; ekran kontrolleri eklenir. | Mevcut gereksinim; kullanıcıya sunulan ürün metni için BL-002 onay kapısı korunur |
| 5 | Entegrasyon hata yolları: `backend/app/providers/market_data.py`, `modules/data_fetcher.py`, `modules/sentiment_analyzer.py` | Başarı, timeout, retry tükenmesi ve kısmi kesinti kontrollü testlerle doğrulanır; mevcut `timeout_seconds` alanının gerçek çağrıya iletilmesi ayrıca incelenir. | BL-003; timeout/retry politikası ve yeni gözlem aracı kararı onaysız alınmaz |

M2'nin alarm/snapshot akışları bu listenin bitmesiyle otomatik kapanmaz.
Kalıcı şema/migration, üretim kimlik doğrulama, SMTP/vendor, deployment ve maliyet
değişiklikleri için AGENTS.md onay kapıları geçerlidir. Constitution hâlâ Draft.

## M1 sonrasında kalan aşamalar ve kapanış ölçütleri

Bu tablo mevcut ürün/ADR hedeflerini toplar; teknik tasarımı veya açık ürün
kararını onaylanmış saymaz. Başlamadan önce ilgili ayrıntılı plan netleştirilir.

| Aşama | Kalan iş | Kapanış kanıtı / onay |
|---|---|---|
| M1 / BL-001 | Patern/sinyal yanında ML yardımcıları, MACD/Bollinger değişken seri testleri | Sabit sentetik veriyle bağımsız beklenen sonuçlar; ağsız testler geçer. Finansal hesap değişikliği ayrıca değerlendirilir. |
| M2 / davetli erişim | Demo token akışından gerçek davet + e-posta link operasyonuna geçiş | Davetli/davetsiz, kullanılmış/süresi dolmuş link ve oturum senaryoları doğrulanır; güvenlik/gizlilik, SMTP/vendor ve operasyon sahipliği açık onay ister. |
| M2 / cockpit ve paper | Auth/hata/boş veri/yeniden deneme, watchlist ve paper akışlarının kapanışı; desteklenen sembol kümesinin doğrulanması | Temel akışlar kontrollü API/tarayıcı testleriyle geçer; BIST100 listesinin güncelleme kaynağı ve sorumlusu belirlenir. Gerçek emir veya muhasebe eklenmez. |
| M2 / alarm ve snapshot | In-app + SMTP bildirimleri ve ADR-0002 hedefindeki günlük snapshot iş hattı | Başarı/hata/tekrar çalıştırma, idempotency ve veri tazeliği kanıtı; worker/queue veri modeli, saklama ve servis ayarları uygulamadan önce onaylanır. |
| M3 / kalıcılık | Auth/watchlist/paper ve iş kayıtları için PostgreSQL geçiş tasarımı | Şema/migration ve veri saklama onayı; geçiş ve geri dönüş testleri; süreç yeniden başlayınca korunması gereken kayıtlar korunur. |
| M3 / dağıtım | Hostinger/Coolify API/web/worker ortamları ve sürüm akışı | Maliyet/vendor, secrets, veri lisansı ve operasyon sahibi onayı; staging ana akış, health, deploy ve rollback denemeleri. |
| M3 / kurtarma ve gözlem | Backup/retention, izole restore testi, sağlayıcı hata/gecikme görünürlüğü, runbook | Restore ve rollback kanıtları kaydedilir; veri sahibi, saklama/şifreleme kararları, alarm eşikleri ve sorumluları onaylanır. |
| CDSK / BL-004–007 | README placeholder'ları, desteklenen Python/Node ve dependency politikası, legacy modülerleştirme planı, izlenen artifact/veri sahipliği | Belgeler kodla eşleştirilir; destek matrisi ve gerekirse ADR onayı alınır; kullanıcı verisi veya artifact'ı onaysız silinmez. |
| CDSK / yönetişim | Constitution TBD'leri, ürün başarı ölçütleri ve açık kararlar | Ürün sahibi açık onayı; gerekli Proposed ADR'ler hazırlanır. Eski belgelerdeki hedef/mevcut durum çelişkileri sessizce kapatılmaz. |
| Git / teslim | Bekleyen değişikliklerin gözden geçirilmesi ve sürümlenmesi | Kullanıcı commit/push onayı verdi; bu teslim için dosya kapsamı ve temel secret desenleri kontrol edilir, push sonrası uzak ref doğrulanır. Bu işlem ürün kilometre taşlarını kapatmaz. |

Bilinen belge borcu: operations `RUNBOOK.md` API başlangıcını hâlâ hedef olarak
anlatıyor; `OBSERVABILITY.md` health endpoint yok diyor, oysa yerel API'de var.
`RELEASE_PLAN.md` yalnızca başlangıç belge teslimini anlatıyor. Üretim hazırlığı
öncesinde bunlar mevcut/hedef ayrımıyla güncellenmelidir; bugün üretime hazır
kanıtı olarak kullanılmamalıdır. Bağımlılık güvenlik riskleri ve LibreSSL
uyarısı ayrıca doğrulanıp ele alınmalıdır; bu kayıt yeni güvenlik taraması değildir.
