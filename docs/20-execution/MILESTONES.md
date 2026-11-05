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
- **Evidence (2026-11-05):** 35 backend testi geçiyor; gerçek teknik hesaplayıcı
  artık doğrudan ve API üzerinden ağsız sınanıyor. Patern/sinyal/ML kapsamı,
  veri niteliği görünürlüğü ve entegrasyon kapanışı hâlâ eksik.

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
