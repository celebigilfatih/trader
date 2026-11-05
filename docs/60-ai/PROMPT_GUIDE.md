# Prompt Guide

## New Project Trigger

> Bu proje CDSK standardını kullanacak. Proje adı: [ad]. Çözdüğü problem:
> [problem]. İlk hedef: [hedef].

## Project-specific Prompt Rules

- Önce veri kaynağı, sembol, periyot, interval ve veri zamanını belirt.
- “Canlı”, “gerçek zamanlı”, “AI” ve “başarı” iddialarını kod/entegrasyon
  kanıtıyla doğrula; doğrulanamıyorsa deneysel veya `TBD` de.
- Mock sentiment ya da sentetik OHLCV kullanılıyorsa bunu açıkça yaz.
- Finansal çıktıyı yatırım tavsiyesi, garanti veya emir talimatı olarak sunma.
- Yeni entegrasyon, deployment, veri modeli veya finansal otomasyon için önce
  onay kapısı ve ADR ihtiyacını değerlendir.

## Prohibited Prompt Content

Secret, doğrudan kişisel veri ve doğrulanmamış iddialar prompt'a kalıcı bağlam
olarak yazılmaz.
