# OS-01 — Stochastic API sözleşme incelemesi

- **Status:** Tasarım tamamlandı; [ADR-0009](../50-decisions/ADR-0009.md) Proposed,
  özel eşleme onayı bekleniyor. Runtime/test/config dosyaları değiştirilmedi.
- **Başlangıç:** `8dd5ddd`, `codex/m1-pattern-signal-tests`, temiz çalışma ağacı.
- **Amaç:** M1-P0-01 OS-01 için seçici → seri anlamını somut ve incelenebilir yapmak.

## Kök neden ve çağıran incelemesi

- Gerçek `add_indicator('stoch')` K/D serilerini üretir; route aynı `stoch`
  anahtarıyla erişir. Hata hesaplamadan sonraki API projeksiyonundadır.
- API okuyucusunun yerel `latest_indicators` sözlüğü `Dict[str, Optional[float]]`
  olarak tanımlıdır; seçilen ad başına skaler/null üretir. Route dönüş annotation'ı
  `Dict[str, object]` olup ayrıca response_model tanımlı değildir. Mevcut MACD
  ana seri örneği çalışır; diğer MACD serileri otomatik dönmez.
- Eski API double'ı her seçici için aynı adlı seri ürettiğinden bu uyumsuzluğu
  örter. Düzeltme regresyonunda gerçek analyzer zorunludur; provider sınırı fixture olur.
- Mevcut cockpit varsayılan seçicilerle çağırır ve yanıtı JSON gösterir;
  `stoch` veya iki Stochastic çizgisi için özel okuyucusu bulunmadı. Core/legacy
  serilere alias eklemek veya frontend'i değiştirmek bu dar düzeltmeye gerekmez.

## Bu adımda çalıştırılan kanıt

Gerçek FastAPI ASGI akışı ve hesaplayıcı, yalnız provider ve servislerin test
state'i kontrollü; auth gerçek test servisidir. Dış socket'ler engelli, Windows
asyncio loopback serbest; TestClient socketpair nedeniyle sandbox dışında koşuldu.
18 barlık sabit fixture: ilk bar H=200/L=0/C=100, ardından H=120/L=80;
son beş kapanış 80/100/120/90/110. Gerçek K=75, D=200/3.

| İstek / seçici | Gözlenen HTTP | İlgili sonuç |
|---|---|---|
| `stoch` | 500 | Eksik `stoch` seri anahtarı; önceki doğrudan route reproducer'ı KeyError verir. |
| `williams_r` | 200 | `-25.0` |
| `cci` | 200 | `null` (18 bar < mevcut20 warmup) |
| `ema_21` | 200 | Sayısal EMA; mevcut yol çalışır. |
| `stoch_k`, `stoch_d` | 400 | Query seçicileri desteklenmez. |
| Kimliksiz `stoch` | 401 | Mevcut auth sınırı çalışır. |

Mevcut API14 + momentum17 = **31 test / 0.237 saniye / OK**, aynı dış socket
engeli altında yeniden geçti. Bu passing paket OS-01'in giderildiği kanıtı
değildir; 500 gözlemi normatif passing test beklentisine dönüştürülmedi.
Tam366 test/34-mutant kaydı önceki momentum adımının tarihsel kanıtıdır;
bu belge adımında yeniden koşulmadı. Starlette/httpx uyarısı sürer.

## Somut öneri ve kabul ölçütleri

Öneri A: `data.indicators.stoch` son %K'nın skaler/null gösterimidir. %D ve yeni
alan/endpoint eklenmez. Gerekçe ve alternatifler ADR-0009'da tek kaynaktır.
Alan adı ve türü korunsa da K/D anlamı daha önce tanımlanmadığından bu ilk
semantik eşleme AGENTS sözleşme/onay kapısı kapsamında değerlendirildi.

Onay sonrası uygulama için kontrol listesi:

- Gerçek analyzer ve HTTP route: `stoch` isteği 200 ve tam `{"stoch":75.0}`
  indikatör sözlüğü verir; D=200/3 seçilmez, ek K/D alanı çıkmaz.
- Geçerli K=0, 0.0 kalır; K=NaN warmup/flat örneğinde anahtar mevcut ve null'dır.
  Fixture provider ile kısa seri sınanması provider'ın en az bar politikasını değiştirmez.
- Karışık seçici isteğinde mevcut diğer sayısal/null değerler, istek anahtarları,
  fiyat ve metadata/correlation bilgisi korunur; istenmeyen seri eklenmez.
- Yeni query adı açılmaz; `stoch_k/stoch_d` 400, kimliksiz istek 401 kalır.
- Önce 200/%K beklentisi mevcut HTTP 500 nedeniyle assertion failure üretir;
  dar API eşlemesinden sonra green olur. `.get(..., None)` genel fallback'i yoktur.
- API/momentum hedefi, tam dış ağ engelli backend, compile/CDSK yapı/link/diff
  ve bağımsız uygulama incelemesi tamamlanır. Finansal etkinlik veya genel M1 kapanışı iddia edilmez.

## Karar ve teslim sınırı

Yeni eşleme henüz uygulanmadı; OS-01 açık, ADR-0001–0008 Accepted kalır.
AGENTS'in “Veri modeli/migration ... açık kullanıcı onayı” kapısı, burada
seçiciye ilk kez verilen K/D anlamı için uygulanıyor; bütün bug fix'lere yeni
onay gerektiği şeklinde yorumlanmıyor. Genel devam talebi özel A onayı değildir.
Sonraki adım A onaylanırsa dar TDD uygulaması; ardından VWEMA/Supertrend ve aktif okuyucu testleri.

Frontend build/browser, canlı Yahoo/RSS/SMTP ve uzak CI çalışması bu belge
adımında yeniden doğrulanmadı. CDSK upstream validator ortamda yok; yapısal
kontroller onun başarı sonucu değildir. Yeni servis/maliyet/vendor veya secret yok.

## İnceleme ve belge doğrulaması

Bağımsız kaynak/fixture/çağıran incelemesi kullanıcı kararını engelleyen bulgu
bulmadı. Yerel sözlük annotation'ının route/OpenAPI imzası sanılmaması için
ifade düzeltildi. Reviewer testleri bağımsız yeniden çalıştırmadı;31-test koşusu
controller kanıtıdır.63 Python compile,39 zorunlu CDSK dosyası,11 Boot bölümü,
8 Accepted/1 Proposed ADR ve değişen belgelerin yerel bağlantıları doğrulandı.
Git diff runtime/test/CI değişikliği olmadığını doğruladı; upstream validator
başarısı iddia edilmez. Belge teslimi, ADR-0009'un kabulü veya OS-01 düzeltmesi değildir.
