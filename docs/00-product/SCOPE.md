# Scope

## MVP In Scope

- Kapalı beta günlük/haftalık karar destek kullanıcıları için Streamlit miras UI + yeni
  `daily decision cockpit` hedefine geçiş hazırlanışı.
- Teknik analiz, tarama, haber/duygu ve deneysel AI çıktılarının tek ekran mantığında
  erişimi.
- BIST100 + `watchlist` symbol setinde OHLCV analizi ve temel sinyal üretimi.
- Paper/simülasyon portföyü (sadece sembol, adet, ortalama maliyet).
- İn-app + e-posta alarm altyapısının kurulumu.
- Gün sonunda otomatik snapshot iş hattı.

## Explicit Non-goals

- Aracı kurum entegrasyonu, emir iletimi veya otomatik alım-satım.
- Lisanslı yatırım danışmanlığı ya da getiri garantisi.
- Çoklu kullanıcı yönetimi, gelişmiş yetkilendirme, otomatik işlem ve broker entegrasyonu.
- “Gerçek zamanlı” veri garantisi; veri üçüncü taraf sağlayıcıya bağlıdır.
- Mock veriyi gerçek piyasa gözlemi gibi sunmak.
- Gerçek nakit, commission, realized PnL/vergisel muhasebe (paper portföyde tutulmaz).

## Later

- Yetkinlik bazlı erişim ve admin iş akışı (davet + onboarding).
- UI’nın `Next.js` kabuğuna tam geçişi ve API-first altyapı.
- Full deterministic test paketi ve veri sözleşmesi.
- Hostinger/Coolify/PostgreSQL dağıtım modeli ve SLA planı.

## Scope Change Policy

MVP genişlemesi change request ve açık kullanıcı onayı gerektirir; mimari etkide
ADR hazırlanır.
