# Agent Operating Protocol

Bu projenin kalıcı bağlamı sohbet değil, sürümlenen repository belgeleridir.

## CDSK başlangıç akışı

Kullanıcı “Bu proje CDSK standardını kullanacak” dediğinde:

1. Repository ve mevcut değişiklikleri incele.
2. Bu protokolü, anayasayı ve Project Boot'u oku.
3. Proje adı, problem ve ilk hedefi mevcut kaynaklardan doğrula; anlamlı eksikleri
   sor, bilinmeyeni `TBD` bırak.
4. Doküman çelişkilerini, açık soruları ve ADR gereksinimlerini çıkar.
5. Onay gereken kararları işaretle.
6. Planla, uygula, doğrula, belgeleri ve handoff'u güncelle.

Kullanıcıdan okuma sırasını her görevde tekrar söylemesini isteme.

## Zorunlu okuma sırası

1. `AGENTS.md`
2. `docs/00-product/CONSTITUTION.md`
3. `PROJECT_BOOT.md`
4. `README.md`
5. `docs/00-product/PRODUCT_SPEC.md`
6. `docs/00-product/SCOPE.md`
7. `docs/00-product/ROADMAP.md`
8. İlgili mimari belgeler
9. Kabul edilmiş ilgili ADR'ler
10. İlgili execution, quality ve operations belgeleri
11. `CHANGELOG.md`
12. Değiştirilecek kod ve testler

## Yetki hiyerarşisi

`CONSTITUTION` > kabul edilmiş ADR > ürün/kapsam > mimari > operasyon/kalite >
`PROJECT_BOOT` özeti > `README`/notlar > sohbet.

Çelişkiyi sessizce çözme; raporla ve kaynakları uyumlu hale getir.

## Onay kapıları

Mimari, teknoloji, dış servis, veri modeli/migration, güvenlik/gizlilik, MVP
kapsamı, maliyet/vendor bağımlılığı ve anayasa değişikliği açık kullanıcı onayı
gerektirir. Gerekirse önce `Proposed` ADR hazırla.

## Operasyon kuralları

- Kullanıcı değişikliklerini koru; destructive işlem yapma.
- Bilinmeyen gereksinim, sürüm veya komutu tahmin etme.
- Secret ve hassas veri yazma.
- Küçük, doğrulanabilir ve geri döndürülebilir adımlar kullan.
- Değişikliğe uygun testleri ve kalite kapılarını çalıştır.
- İlgili belgeleri, `CHANGELOG.md` ve Project Boot AI Handoff bölümünü güncelle.

## Kapanış raporu

Yapılan iş, değişen dosyalar, doğrulamalar, kararlar/ADR'ler, açık riskler ve
önerilen sonraki adımı bildir.
