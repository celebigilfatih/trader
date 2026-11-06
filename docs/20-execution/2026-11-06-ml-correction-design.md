# ML-01–03 dar düzeltme tasarımı ve karar hazırlığı

- **Date:** 2026-11-06
- **Status:** Tasarım aşaması kaydı; sonraki açık onayla ADR-0007 Accepted ve uygulandı.
  Güncel kanıt: [execution planı](2026-11-06-adr0007-implementation.md).
  Aşağıdaki öneri/onay bekleme ifadeleri uygulama öncesi tasarım aşamasına aittir.
- **Başlangıç:** HEAD `6ed1a59`, `codex/m1-pattern-signal-tests`; önceki dirty
  değişiklikler korunur. Bu dilimde üretim kodu/test/dependency/artifact değişmez.
- **Kaynak:** [ML test kaydı](2026-11-06-ml-helper-tests.md),
  [BACKLOG](BACKLOG.md), [önerilen sözleşme](../50-decisions/ADR-0007.md).

## Amaç ve tercihler

Geleceği olmayan satırı etiketli saymamak; eğitim/test/tahminde aynı ön işleme ve
hedef dönüşümünü kullanmak. Kullanıcının “Sıradaki Adım” isteği önceki handoff'taki
düzeltme tasarımını ilerletir; yeni kalıcı format veya mimari onayı sayılmaz.

Seçenekler: quantile kırpmayı kaldırmak (az durum, mevcut kural kaybı), train-only
bounds (önerilen; aynı quantile'lar, öğrenilmiş durum saklama gereği), yeni pipeline/
rolling yaklaşım (daha geniş refactor/politika). Önerilen seçenek ajan tercihidir;
kullanıcı tarafından henüz seçilmedi. Somut kuralların tek kaynağı ADR-0007'dir.

## Root cause / güncel kanıt

| Kusur | Kaynak | Bu adımda tekrar gözlenen |
|---|---|---|
| ML-01 | ml_predictor.py target fillna(0) → train valid mask | Close100/110/121/133.1, h2: [.21,.21,0,0]; gerçek bilinmeyen son iki hedef kayboluyor. |
| ML-02 | Tüm fit scaled; predict yalnız svr scaled/inverse | Gerçek LR + bağımsız .025 etiket sınırında Close160→160, beklenen164; alınan X scaled X'e eşit değil. |
| ML-03 | prepare_features temizler; train tekrar temizler; her çağrı sample quantile | raw[0,10,20] ilk satır.2; sona10000 ekleyince.3. |

40 mevcut ML testi dış socket/joblib engelli tekrar geçti (1.013 s). Bu testler
kusurların kapandığını iddia etmez; yeni üretim davranışı uygulanmadı.
Kaynak taraması aktif legacy çağrının train/predict'e aynı ufku verdiğini, eğitim
hatasında döndüğünü gösterir. save/load için uygulama içi consumer bulunmadı;
repository dışı artifact varlığı bilinmiyor, hiçbir kullanıcı dosyası açılmadı.

## Onaydan sonra yapılacak dilimler / kabul ölçütleri

Bu bir tasarım ve kabul sırasıdır; ayrıntılı uygulanabilir görev planı onaydan
sonra hazırlanır. Şu anda RED testleri veya üretim kodu yazılmaz.

1. **Hedef ve zaman sınırı:** h1/2/3 literal prefix + NaN tail; gerçek sıfır
   getiri korunur; h>=N/empty/invalid horizon ve nonfinite/pozitif olmayan fiyatlar.
   Native hedef kullanılarak 49/50 geçerli etiket sınırı ve index/caller korunması.
   N53,h3 ile50 etiket, varsayılan split40/10, purge sonrası37 train/10 test.
   Tutulan her train satırında i+h<ilk test kaynak konumu; yetersiz purge hata.
2. **Sabit temizleme ve train-only bounds:** prepare_features geçmiş prefix'i
   sona veri eklenince aynı; cleaning eski nonfinite/ffill/sabit yüzde kuralını
   korur. Raw train[0,10,20] için q01=.2, q99=19.8, holdout10000→19.8; holdout
   bounds/scaler fit'ini değiştirmez (bölünme noktası sabit tutulur). Sabit kolon
   sınır almaz, caller/index/kolon sırası korunur. İkinci quantile pass yoktur.
3. **Ortak inference:** dört gerçek estimator için alınan X, training scaler'ın
   aynı feature sırasındaki transform çıktısı; model çıktısı inverse target'tan
   sonra mevcut cap'e gider. Bağımsız sabit .025 hedef sınırında fiyat160→164
   literal oracle; bunun target boundary double'ı olduğu belirtilir. Ayrıca
   native hedefle gerçek dört-model fit→predict smoke'u; accuracy oracle'ı değildir.
   Eksik feature/ufuk uyuşmazlığı error, geçerli fiyat sözlüğü ve ensemble aynı biçim.
4. **Tutarlı state/artifact:** failed fit ve fit-sonrası metric hatalı model
   pakette yok; partial başarı yalnız yeni successful modelleri içerir; tamamen
   failed retrain ardından predict/save hata. Güvenilen temporary joblib roundtrip
   sınır/kolon/scaler/ufuk/tahmini korur. Sürümsüz/bilinmeyen/bozuk paket load
   ValueError, var olan instance'a mutation yok; kullanıcı artifact'ı kullanılmaz.
5. **Birleşim:** değişen sayısal contract testleri yeni ADR oracle'ına geçirilir;
   eski hatayı korumak için üretim düzeltmesi gevşetilmez. Gerçekçi mutation/RED
   kanıtı, bağımsız review, tüm backend dış ağ engelli ve kontrollü legacy render.
   Compile, CDSK yapı/bağlantı/diff, docs/changelog/handoff; canlı sağlayıcı ayrı.

## Etki, açık sınırlar ve onay

- Dokunulacak ana üretim dosyası `modules/ml_predictor.py`; mevcut üç ML test
  dosyası yeni sözleşmeye uyarlanır, hedefli regresyon/persistence testleri eklenir.
  app.py yalnız gerçek çağrı/diagnostic uyumu gerekirse dar biçimde ele alınır;
  yeni ekran/servis/vendor/dependency/config veya MVP genişlemesi yoktur.
- Target Series, purge sonrası sample sayısı ve joblib formatı bilinçli değişir.
  Eski artifact otomatik migrate edilmez; yeniden eğitim gerektirir. Yeni bounds
  ile eski model karıştırılmaz. Advanced ML-04 bu tutarlılığın gerekli parçasıdır;
  SimpleMLPredictor ve ML-05–08 düzeltmeleri ayrı kalır.
  Eksiksiz girdide50 etiket için en az50+h ham gözlem gerekir (h30→80);
  mevcut kısa veriyle daha çok yetersiz eğitim hatası görülebilir.
- Tarih ufku halen gözlem sayısıdır; UI takvim günü farkı açık. Caller indicator
  causal garantisi, performans/kalibrasyon, merkezi registry ve M1 kapanışı yoktur.
- AGENTS.md: “Mimari, teknoloji, dış servis, veri modeli/migration, güvenlik/gizlilik,
  MVP kapsamı, maliyet/vendor bağımlılığı ve anayasa değişikliği açık kullanıcı onayı
  gerektirir.” Bu tasarım geçici hedef/ön işleme ve kalıcı model paketi sözleşmesini
  değiştirir; öneri Proposed kalır. Açık onaydan sonra Accepted ve uygulama planı.

## İnceleme / kapanış kanıtı

- Bağımsız builder reviewer ADR/tasarım ve üretim ML/ilgili çağıranları okudu;
  planlamayı engelleyen ek önemli soru bulmadı. Belgedeki ürün sahibi onayı tek
  kalan karar kapısıdır; reviewer onayı kullanıcı onayı sayılmaz.
- 57 Python dosyası memory compile, 39 zorunlu belge/11 numaralı Boot bölümü;
  Accepted ADR-0001–06/Proposed ADR-0007 ve21 yerel Markdown bağlantısı doğrulandı.
  Değişen9 belgede whitespace/tekrar eden ana başlık ve git diff --check geçti.
- ML üretim modülleri/config/dependency manifestleri HEAD ile aynı; test dosyalarına
  bu adımda yazma yapılmadı. HEAD6ed1a59, commit/push yok. Önceki dirty değişiklikler
  korundu. Upstream CDSK validator macOS yolu ortamda yok; yapısal kontrol onun
  yerine geçmiş sayılmaz. Full backend/frontend/canlı servis yeniden koşulmadı;
  bu adımın runtime kanıtı40 hedef test ve üç ayrı kusur reproducer'ıdır.
