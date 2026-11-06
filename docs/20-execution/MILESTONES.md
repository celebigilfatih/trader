# Milestones

## Milestone — M0 CDSK başlangıcı

- **Outcome:** Mevcut ürünün doğrulanmış bağlamı, riskleri ve çalışma kuralları
  repository içinde kalıcıdır.
- **Dependencies:** CDSK 0.1.0 yerel kit.
- **Exit criteria:** Zorunlu dosyalar var; proje/mimari/operasyon bilgisi kodla
  uyumlu; CDSK validator ve sözdizimi kontrolü geçer.
- **Risks:** Mevcut davranışı belgelerken doğrulanmamış ürün iddiası üretmek.
- **Status:** Completed

## Milestone — M1 güvenilirlik tabanı

- **Outcome:** Ana hesaplamalar ve hata davranışı güvenilir biçimde doğrulanır.
- **Dependencies:** BL-001, BL-002 ve BL-003.
- **Exit criteria:** Çevrimdışı test paketi; açık veri provenance; entegrasyon
  hata görünürlüğü.
- **Risks:** Mevcut UI monoliti ve dış servislerin değişken yanıtları.
- **Status:** In Progress
- **Evidence (2026-11-06):** 349 backend testi dış socket engelli geçti;
  patern/sinyal/risk/OTT/ML ve onaylı RSS/provenance dilimleri doğrulandı.
  Kalan indikatörler, legacy tarayıcı zincirleri ve diğer aktif veri/hata yolları
  genel kapanışı engelliyor. Kabul matrisi ve sıralı P0 işleri
  [M1 kapanış kontrolündedir](M1_CLOSEOUT.md); sonraki dilim Stochastic/Williams
  %R/CCI testleridir. Önceki 35 test kaydı tarihsel başlangıç kanıtıdır.

## Milestone — M2 ürün doğrulama ve çekirdek açılımı

- **Outcome:** Kapalı beta onboarding’i, paper portföy ve karar cockpit’inin temel
  akışları doğrulanır.
- **Dependencies:** ADR-0001, ADR-0002, ADR-0003.
- **Exit criteria:** Davet/kimlik akışı, günlük snapshot, alarm + bildirim, symbol
  universe (BIST100 + watchlist) stabil çalışır.
- **Risks:** API geçişinin Streamlit ile uyumluluk ve veri sürekliliği etkisi.
- **Status:** In Progress

## Milestone — M3 üretime hazır mimari

- **Outcome:** FastAPI + Next.js + PostgreSQL + Coolify dağıtım hattı devreye alınır.
- **Dependencies:** Veri lisansı, maliyet modeli, monitoring ve backup onayı.
- **Exit criteria:** Health endpoint, rollback, backup restore testi, CI gate.
- **Risks:** Yetkilendirme ve operasyon ekiplemelerinde gecikme.
- **Status:** Proposed
