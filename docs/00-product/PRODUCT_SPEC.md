# Product Specification

## Problem

BIST hisseleri için fiyat verisi, teknik göstergeler, tarama sonuçları, haberler
ve deneysel tahminler farklı araçlarda dağınıktır. Proje bu verileri tek bir
yerel Streamlit arayüzünde bir araya getirerek araştırma süresini azaltmayı
amaçlar.

## Target Users and Outcomes

- **Birincil kullanıcı:** BIST hisselerini kendi adına araştıran bireysel
  kullanıcı veya analist.
- **Beklenen sonuç:** Seçilen sembol ve zaman aralığı için veriyi, göstergeleri
  ve sinyalleri aynı ekranda incelemek; uygun hisseleri kurallarla taramak.
- **Kısıt:** Çıktılar karar desteğidir; yatırım danışmanlığı veya emir yürütme
  hizmeti değildir.

## Value Proposition

Teknik analiz, grafik, tarama, haber/duygu analizi ve deneysel makine öğrenmesi
çıktılarını tek bir araştırma arayüzünde sunmak.

## Functional Requirements

- Yahoo Finance üzerinden OHLCV ve şirket verisi çekmek.
- Seçilen indikatörleri hesaplamak ve Plotly grafiğinde göstermek.
- Birleşik AL/SAT/BEKLE sinyalleri ile risk açıklamaları üretmek.
- Teknik kriterlerle çoklu hisse taraması yapmak.
- RSS kaynaklarından haber çekmek ve yerel NLP kütüphaneleriyle duygu skoru
  hesaplamak.
- Geçmiş fiyatlardan deneysel scikit-learn tahmini üretmek.
- Teknik analiz çıktısını PDF olarak indirebilmek.
- Kapalı beta erişimi davet ile sınırlandırmak.
- Paper portföyde sadece sembol, adet, ortalama maliyet bilgisini barındırmak.
- Gün sonunda otomatik tarama ve analiz snapshot'ı üretmek.

## Non-functional Requirements

- Finansal sonuçların kaynağı, zamanı ve deneysel niteliği kullanıcıya açık
  olmalıdır.
- Dış servis hatası uygulamanın tamamını çökertmemeli ve kullanıcıya anlaşılır
  geri bildirim vermelidir.
- Secret değerleri repository'ye yazılmamalıdır.
- Hesaplama çekirdeği ağdan bağımsız, deterministik birim testleriyle
  doğrulanabilir olmalıdır.
- Yerel başlangıç ve kurtarma adımları dokümante edilmelidir.
- Günlük/haftalık karar cockpit’inde geçmiş snapshot karşılaştırmaları izlenebilir
  ve verinin tazeliği raporlanmalıdır.
- Üretim öncesi veri lisansı, cooldown, sağlık metrikleri ve rollback planı
  yazılı kabul edilmelidir.

## Success Metrics

Ürün hedefi ve ölçülebilir başarı metrikleri henüz sahibi tarafından
onaylanmadı. Önerilen ilk metrikler: kritik akış başarı oranı, veri tazeliği,
dış servis hata oranı ve deterministik test kapsamı.

## Acceptance Criteria

- Uygulama yerel ortamda belgelenen komutla açılır.
- Desteklenen bir sembolde veri geldiyse teknik analiz ve grafik üretilir.
- Veri gelmediğinde sahte piyasa verisi gerçek veri gibi gösterilmez.
- Mock/deneysel duygu ve ML sonuçları kullanıcıya açıkça etiketlenir.
- Yatırım tavsiyesi olmadığına dair uyarı görünürdür.
- Çevrimdışı çekirdek testleri ve CDSK doğrulaması geçer.
