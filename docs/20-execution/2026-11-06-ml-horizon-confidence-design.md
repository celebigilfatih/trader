# ML-07/08 ve gözlem ufku tasarımı / Git teslim kaydı

- **Date:** 2026-11-06
- **Status:** Tasarım hazır; [ADR-0008](../50-decisions/ADR-0008.md) Proposed.
- **Başlangıç:** Teslim edilmiş checkpoint `abfc592`; aynı codex dalı.

## Commit & push — tamamlandı

Kullanıcının “Commit & Push yapalım” talebiyle70 dosya `abfc592` commit'ine alındı
ve `origin/codex/m1-pattern-signal-tests` dalına push edildi. Uzak dal yeni açıldı,
upstream ayarlandı; son kontrolde ahead/behind0/0 ve temiz working tree doğrulandı.
Merge/PR/force-push yapılmadı. Git ortamın otomatik yerel committer kimliğini kullandı;
kimlik ayarı değiştirilmedi. Kullanıcının sonraki adım talebi aşağıdaki tasarımı kapsar.

Commit öncesi yeni kanıt:331 backend testi dış socket engelli geçti (24.007 s,
asyncio loopback serbest); Next.js14.2.18 üretim build/lint/type kontrolü başarılı.
Üretilmiş frontend + Edge üzerinde10/10 deterministik senaryo, beklenmeyen dış
istek0. Yerel test sunucusu yalnız bu koşu için açılıp kapatıldı; canlı sağlayıcı yok.
60 Python compile,39 zorunlu belge/11 Boot bölümü/diff kontrolü; yeni credential
kalıbı veya hassas artifact yolu bulunmadı. Bu sınırlı tarama güvenlik audit'i değildir.
Upstream CDSK validator ortamda yok; dependency/deprecation borçları ayrı kalır.

## İnceleme ve reproducer

- predict_price'taki ilk aralık ile ikinci exact/fallback cap çelişiyor.
  Real LR/scaler + bağımsız .40 label boundary h1/2/3/7/8/14/30/31 fiyat/current
  1.10/1.10/1.10/1.25/1.10/1.10/1.40/1.10. Üretim cap'i değiştirilmedi.
- Fiyat sözlüğü get_model_confidence'a doğrudan verilince TypeError. Helper için
  aktif UI caller yok; UI kendi dispersion skorunu hesaplıyor. Algoritmalar ayrı.
- Panel kartı now+h, grafik last_data+h takvim günü kullanıyor. Bayat sentetik
  fixture'da now2026-11-06/last2026-01-08/h3: kart2026-11-09, grafik2026-01-11.
  Bu source/aritmetik kanıtıdır, bu adımda yeni canlı UI render ölçümü değildir.

## Amaç ve seçenek

Öneri: ilk mevcut cap aralıklarını tek yardımcıya almak, helper'a fiyat/current
bağlamı eklerken legacy return kaydını desteklemek, panelde h gözlem ve şematik
adım ekseni kullanmak. Efektif fallback cap'lerini sabitlemek tutarsızlığı kural
yapar; nested price response veya yeni kalibre model/takvim altyapısı işi büyütür.
Önerilen seçenek ajan tercihidir; özel kullanıcı onayı henüz yoktur.

Somut oran/type/error/UI metinleri ve uyum sınırlarının tek kaynağı ADR-0008'dir.
Bu kayıt ayrıntılı implementation planı değildir; onay sonrası plan/RED→GREEN gelir.

## Etki ve kabul ölçütleri

- Ana üretim kapsamı modules/ml_predictor.py ve app.py paneli; yeni public pure
  get_prediction_return_bounds ve optional current_price helper parametresi.
  Price sözlüğü/model/feature/scaler/purge/artifact format1 ve Simple aynı kalır.
- h1/2/7/8/30/31, iki yön, eşitlik ve bütün geçerli integer aralıklarını literal
  fiyatlarla test etmek; tek cap, no fallback. İnverse/scaler/native smoke korunur.
- Fiyat/legacy/helper invalid girdi: context zorunluluğu, bool/string/nonfinite,
  karışık kayıt/error payload, boş/yalnız ensemble, strict .7/.4; mutasyon yok.
- Bayat tarihli kontrollü AppTest kart/başlık/selector/caption ve0..h adım
  grafiğini doğrular; model/fixture doubles ve gerçek fit kanıtı ayrı tutulur.
- UI model-yayılımı formülü/helper confidence karıştırılmaz. Kapsam garanti,
  kalibrasyon/getiri başarısı, piyasa takvimi, yeni provider veya kalıcı şema değildir.
- Rollback birlikte kod+UI; eski cap/helper/tarih kusurları açık kaydedilir.

## Onay / sonraki adım

AGENTS.md mimari/veri modeli/finansal sözleşme etkisi nedeniyle Proposed ADR
hazırlanır. Kullanıcının commit/push isteği bu belgeyi paylaşmak için yetkilidir;
önerilen cap/helper/UI kuralının uygulanmasına özel onay sayılmaz.
Kalan kapı ADR-0008'in açık onayı; üretim kodu ve testler bu tasarım adımında aynı.

## İnceleme / belge doğrulaması

- Bağımsız builder reviewer ADR/tasarım/üretim kodu ve ilgili testleri okudu;
  implementation planı için önemli açık soru bulmadı. Kalan karar özel ürün sahibi
  onayıdır; reviewer onayı Accepted kararı sayılmaz.
- ADR dizini, AI_ARCHITECTURE, DATA_MODEL, BACKLOG, TEST_STRATEGY, CHANGELOG ve
  Boot1.49 bu Proposed durumu/teslim kanıtıyla eşleştirildi. Runtime/test/model/
  config/dependency checkpoint'ten değişmedi; yalnız dokümantasyon diff'i vardır.
- Compile60, manifest39 belge/11 Boot bölümü; ADR1–7 Accepted/ADR8 Proposed,
  34 yerel link/whitespace/diff ve production değişmezliği son kontrolde doğrulandı.
  Bu öneri belgeleri kullanıcının aynı commit/push yetkisiyle ayrı docs commit'inde
  paylaşılır; uygulanmış finansal politika veya yeni sürüm artifact'ı değildir.
