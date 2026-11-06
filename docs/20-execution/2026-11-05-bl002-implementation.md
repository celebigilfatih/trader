# BL-002 uygulama planı

> Agentic workers: test-driven-development ile her işi doğrula; bağımsız dosya
> alanlarında parallel agents kullanılabilir. Kapanışta tüm diff ayrıca incelenir.

**Goal:** Onaylanan veri niteliği metinlerini ve ADR-0004 RSS/NLP sözleşmesini
mevcut cockpit ve legacy ekranlarına uygulamak.

**Architecture:** Mevcut FastAPI/Next.js ve Streamlit katmanları korunur. RSS
haber sözlükleri Accepted ADR-0004 alanlarını taşır; UI'lar nullable skor/zamanı
tüketir. Yeni sağlayıcı, model, kalıcı şema veya dependency politikası seçilmez.

**Tech Stack:** Mevcut Python/unittest, pandas, Streamlit, Next.js, Playwright.
**Spec:** `docs/00-product/BL002_DATA_DISCLOSURE.md`, `docs/50-decisions/ADR-0004.md`.

## Ortak sınırlar / inceleme odağı

- Onay: Kullanıcı, metin ve ADR için doğrudan sorulan soruya “devam et” dedi.
- Önceki çalışma ağacını koru; commit/push yok, yeni servis/versiyon/migration yok.
- Gerçek ağ veya kullanıcı verisi unit fixture olmaz. NLP/ML hesap formülleri,
  mevcut sentiment eşikleri ve ML ufuk/tarih anlamı bu işte değiştirilmez.
- Eksik kaynak/zaman, gerçek hesaplanmış sıfır, NLP hatası, farklı UTC offset'leri,
  stale analiz yanıtları ve geçersiz model yayılımı ayrı sınanır.
- Başlangıç: HEAD `6ed1a59`, dal `codex/m1-pattern-signal-tests`; önceki 60 backend
  testi ve 5 UI senaryosu kayıtlıdır, kapanışta yeni toplam yeniden doğrulanır.

## Task 1 — RSS/NLP sözleşmesi (bağımsız dosyalar)

**Files:** `modules/sentiment_analyzer.py`, yeni `backend/tests/test_sentiment_analyzer.py`.
**Interface:** ADR-0004 NewsItem: `published_at`/`published_time_status`/UTC
`fetched_at`, `date` geçersizde boş; `sentiment_status`, nullable skor/etiket.
Özet `scored_news`, `failed_sentiment_count`; skorsuzda nullable ortalama/etiket.

- [x] Eksik/geçersiz/offset'li tarih, gerçek nötr, NLP hatası, sıralama, karma/boş
  özet ve sembol filtre regresyonlarını yaz; eski kodda anlamlı hatayı doğrula.
- [x] Sözleşmeyi uygula; unit testlerini ağ kapalı koşuda geçir.
- [x] Yeni alanları eski kullanıcı verisi/secret/log metni saklamadan doğrula.

## Task 2 — Legacy görünürlük ve nullable okuyucular

**Files:** `app.py`, gerekirse dar saf sunum helper'ı, yeni legacy testleri.
**Consumes:** Task 1 sözleşmesi (tam alanlar ADR-0004'te).
**Produces:** RSS gerçek/deneysel ayrımı ve doğru yayın/çekilme zamanı; nullable
skor filtre/özetleri; deneysel ML metinleri ve geçersiz yayılım gösterim guard'ı.

- [x] Kontrollü legacy senaryolarını yaz; önce eski davranışta hatayı doğrula.
- [x] Onaylı N/M metinlerini uygula; sabit olasılıkları kaldır, geçersiz yayılımda
  ±%5 fallback'i gösterme. Model/formül/ufuk veya sinyal hesabını değiştirme.
- [x] Gerçek nötr / başarısız NLP / haber yokluğu / filtreler, ML üst uyarısı,
  model uyumu ve geçersiz bant durumunu kontrollü verilerle doğrula.

## Task 3 — Cockpit ve deterministik browser

**Files:** `frontend/app/page.js`, `scripts/frontend_ui_scenarios.mjs`.
**Consumes:** Mevcut analiz `data` + `meta` ve RSI satır kaynak/zaman alanları.

- [x] Yeni beklenen metadata, demo/unknown/eksik kaynak, boş/hata metinleri ve
  stale analiz senaryolarını yaz; mevcut ekranda başarısız olduklarını gör.
- [x] Onaylı G/D/R/A metinlerini uygula; metadata'yı koru. Yeni istek/sembol/
  hata/logout sırasında eski veri veya geç gelen eski yanıt güncel görünmesin.
- [x] Mevcut RSI tercih, watchlist, portföy akışlarını koruyarak browser setini geçir.

## Task 4 — Birleştirme / kapanış

- [x] Backend test paketi, ilgili kontrollü legacy testleri, frontend build ve
  deterministik browser setini çalıştır. Canlı sağlayıcı testi ayrı raporlanır.
- [x] Bağımsız diff incelemesini tamamla; önemli bulguları testle düzelt.
- [x] CDSK yapı/bağlantı, sözdizimi ve diff kontrolü; mimari/veri modeli, backlog,
  quality, changelog ve Project Boot handoff güncellenir.
- [x] Model ufuk/tarih, sentiment eşik uyumu, BL-001/003 ve üretim riskleri açık
  bırakılır; M1/M2 tamamlandı iddiası yapılmaz.

## İlerleme / karar kaydı

- Onay alındı ve ADR-0004 Accepted olarak işaretlendi. Sözleşme tüketicileri ile
  aynı teslimde uygulanır; B yalnız etiket değişimi sayılmaz.
- Dosya sahipliği: RSS worker yalnız sentiment modülü/testleri; legacy worker
  yalnız app/helper/testleri; ana ajan frontend/browser ve belgeler.

- Task 1: 14 RSS/NLP testi; eski kodda 17 assertion, ek nonfinite/offset red
  görüldü; sonlu skor ve +03:00 uyumu green. Her RSS testinde socket engeli var.
- Task 2: 8 gerçek Streamlit AppTest; ilk red 12 assertion, ek geçersiz uyum
  guard'ı üç subtest red; son birleşik paket 82/82 green. Tam monolit/browser değil.
- Task 3: 10/10 üretim browser senaryosu. İlk red eksik genel uyarı; inceleme
  sonrası bozuk metadata 0 yerine 3 bilinmeyen alan beklentisinde red; tip guard'ı
  sonrası green. Son Next.js üretim build başarılı.
- Task 4: Bağımsız incelemede Important/Critical bulgu yok. Küçük metadata tipi
  önerisi testle düzeltildi. Son 82 test dış ağ engelli geçti; asyncio loopback
  serbest. 41 Python ve iki Node sözdizimi kontrolü geçti.
- Açık sınırlar: canlı sağlayıcı veya tam legacy browser yok; CDSK upstream
  validator bu ortamda yok. Next.js güvenlik riski, deprecation, ML ufuk/tarih
  ve sentiment eşik borçları sürüyor. Commit/push veya yeni manifest yok.
