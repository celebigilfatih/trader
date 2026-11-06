# VST-01–03 düzeltme sözleşmesi tasarımı

- **Status:** Tarihsel tasarım kaydı. O aşamada ADR-0010 Proposed idi;
  sonraki açık onayla Accepted ve [uygulandı](2026-11-06-adr0010-implementation.md).
  Aşağıdaki gözlem ve doğrulamalar tasarım aşamasının tarihsel kanıtıdır.
- **Başlangıç:** `698461b`, `codex/m1-pattern-signal-tests`, temiz çalışma ağacı.
- **Amaç:** Önceki test diliminde bulunan üç açıklık için belirsiz formül ve
  hazır olma seçimlerini incelemeye/onaya hazır hale getirmek.

## Kök neden ve yeni kanıt

| Bulgu | Bu adımda tekrar gözlenen | Öneri A'nın kapatacağı davranış |
|---|---|---|
| VST-01 | Dört100'de VWEMA5 bar0=100; beşinci200 eklenince bar0=120. | Gelecek pencere seed'i yerine yalnız ilk pozitif hacimli gözlem ve geçmiş durum. |
| VST-02 | Beş100 sonrası130'un hacmi1 veya1000 iken sonuç110. | Fiyat*hacim ve hacmin ayrı üssel durumları; ağırlık sadeleşmez. |
| VST-03 | Supertrend9 barında yalnız yön1/bull_score1.5;10 barın bar9 fiyatı94,11 barın bar9'u NaN. | Sabit ilk hazır konum p; n<=p'de iki alan da NaN, fiyatı olmayan yön yok. |

Mevcut27 VWEMA/Supertrend+API testi dış socket engelli **0.256 s /OK** yeniden
geçti; asyncio loopback serbest. Bu düzeltme kanıtı değildir; bug karşı örnekleri
normatif passing beklentiye çevrilmedi. Tam386/32-mutant önceki dilimin tarihsel
kanıtıdır ve bu yalnız tasarım adımında yeniden koşulmadı.

Yeni beklenen weighted örnekler bağımsız tam rasyonel aritmetikle kontrol edildi:
p2 için100,1300/7,100; p5 hacim1000 etkisi21700/167; sıfır hacimden sonraki
pozitif bar650/7. Bunlar henüz üretim fonksiyonunun çıktısı değildir.

## Çağıran ve uyumluluk envanteri

| Yol | İncelenen etki / kabul kanıtı |
|---|---|
| `TechnicalAnalyzer.add_indicator/get_latest_indicators` | Aynı index/uzunluk/alanlar; hazır olmayan NaN getter'da eksik kalır. Empty/invalid/atomic ve prefix testleri. |
| Analiz API'si | `vwema_5` varsayılandır; aynı adlarda scalar/null. Gerçek analyzer+fixture provider ile yeni ağırlık değerleri/null, metadata ve auth korunur. `supertrend` query'si yalnız fiyatı döndürmeye devam eder. |
| AlertSystem sinyal/ayı/pozisyon okuyucuları | Eksik yön ==±1 sayılmaz;9/10 barda0 katkı,11 barda mevcut seed ve sonraki dönüşler. BIST fetcher'ın15 bar alt sınırı değiştirilmez; doğrudan kısa çekirdek örneği canlı UI olayı sayılmaz. |
| Legacy chart ve VWEMA5 kartı | Chart mevcut Series'i çizer; NaN geriye doldurulmaz. Kart bugün NaN'i ₺nan ve Altı gösterebilir; A paketinde —/Hesaplanamadı ve nötr durum. Kontrollü AppTest gerekir. |
| MLPredictor optional teknik özellikleri | VWEMA EMA-ratio ve Supertrend feature olarak tüketilebilir. Eski matematikle üretilmiş bu özelliklerle eğitilen ilgili model yeniden eğitim ister; format1 değişmez/otomatik tespit iddiası yoktur. Mevcut AI paneli RSI/EMA/VWAP/Bollinger/MACD seçer; bu iki göstergeyi eklememektedir. |

## Seçim ve kapsam

A paketi üç bulguyu birlikte çözer; yalnız Supertrend ile başlamak daha küçük
ama VWEMA açıklıklarını açık bırakır. Önerilen formül, input-domain, zero/empty,
seed ve rollout/rollback tek kaynak olarak ADR-0010'da tanımlıdır.
Bu seçim yeni sağlayıcı veya finansal etkinlik iddiası değildir. p10 için ilk
hazır konum10 seçimi, p-1'e geçip uzun seri davranışını genişletmekten kaçınır.
ATR/Wilder ve mevcut rolling fallback ayrımı korunur; yeni trailing algoritma yoktur.

Onay sonrası doğrulama: bağımsız RED→GREEN core testleri, volume sensitivity/
prefix/leading-zero/all-zero/p1/invalid-atomic, Supertrend sınırları, gerçek
API ve risk/sinyal zincirleri, eksik VWEMA kartında kontrollü AppTest; mevcut
geçerli13 test ve tam backend/compile/CDSK yapı/link/diff, bağımsız son inceleme.

## Yetki ve teslim

ADR-0010 Proposed'dır; genel devam isteği bu yeni somut matematik/input/UI
paketinin özel onayı sayılmadı. AGENTS “veri modeli/migration ... açık kullanıcı
onayı” kapısı yayınlanan değer/readiness sözleşmesine uygulanır; yalnız test
ve belge değişikliklerine genel yeni onay şartı getirilmez.
Sonraki adım A özel kararından sonra dar uygulamadır. VST-01–03 ve genel M1 açık.

Bu adımda frontend/browser, canlı sağlayıcı, uzak CI ve upstream CDSK yeniden
doğrulanmadı; upstream validator ortamda yoktur. Yeni runtime/dependency/model
formatı/servis/secret veya artifact işlemi yapılmadı.

## İnceleme ve doğrulama

Bağımsız tasarım incelemesi engelleyici matematik/sözleşme bulgusu bulmadı.
Yeniden eğitilmesi gereken modelin eski matematikle eğitilmiş model olduğu
metinde netleştirildi. Oracles, p1/zero/seed, Supertrend hazır olma, atomik
hata, API/reader/kart ve rollout sınırları incelendi. Bu inceleme uygulama,
performans, yeni test başarısı, canonical/finansal etkinlik veya deployment onayı değildir.
65 Python compile/39 zorunlu belge/11 Boot/9 Accepted+1 Proposed ADR ve yerel
bağlantı/diff kontrolü yapıldı; üretim/test değişikliği olmadığı doğrulandı.
