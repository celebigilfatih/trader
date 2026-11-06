# Patern ve sinyal doğrulama planı

> Uygulama: mevcut oturumda sırayla; kaynak kapsam BACKLOG.md içindeki ikinci iş.

**Goal:** Gerçek patern ve sinyal hesaplarını sentetik OHLCV verisiyle ağsız doğrulamak.

**Architecture:** Mevcut unittest keşfine iki test modülü eklenir. Hesaplama
kodları, dış servisler, şema ve bağımlılık bildirimleri değiştirilmez.

**Tech Stack:** Python, unittest, pandas ve mevcut TechnicalAnalyzer.

**Spec:** `docs/20-execution/BACKLOG.md`, BL-001 / sıralı iş 2.

## Kısıtlar ve inceleme odağı

- Gerçek kullanıcı verisi, ağ, SMTP ve rastgele fixture kullanılmaz.
- Beklenen değerler hesaplayıcının çıktısından üretilmez.
- Eşik eşitliği, yönün tersine çevrilmesi, yetersiz veri, nötr sonuç ve
  girdinin korunması sınanır.
- Testler mevcut dar davranışı doğrular; paternlerin finansal geçerliliğini
  veya gelecekteki veriyi kullanmadan çalıştığını kanıtlamaz.
- Hata bulunursa ayrı yeniden üretimle değerlendirilir; finansal davranış
  sessizce değiştirilmez.

## Task 1: Patern regresyonları

**Files:** Create `backend/tests/test_pattern_recognition.py`.
**Interfaces:** PatternRecognition, AdvancedPatternRecognition,
TechnicalAnalyzer.detect_chart_patterns; OHLCV DataFrame -> seri/sözlük/tablo.

- [x] Doji/gölge/saran/yıldız örneklerinde olumlu ve olumsuz sonuçları yaz.
- [x] Sentetik iki tepe/dip, FVG eşik/dolum ve order block yön/güç örneklerini yaz.
- [x] `python -B -m unittest discover -s backend/tests -p test_pattern_recognition.py -v`
  çalıştır; bütün testlerin geçmesi beklenir.

## Task 2: Sinyal regresyonları ve kapanış

**Files:** Create `backend/tests/test_alert_signals.py`; update BACKLOG,
TEST_STRATEGY, OVERVIEW, CHANGELOG ve PROJECT_BOOT.
**Interfaces:** Gerçek AlertSystem, sentetik verili TechnicalAnalyzer -> AL/SAT/BEKLE.

- [x] RSI uç/eşik, kesişim, trend değişimi, Bollinger toleransı, hacim ve
  birleşik sinyal örneklerini bağımsız beklenen sonuçlarla yaz.
- [x] Yeni testleri çalıştır; bütün testlerin geçmesi beklenir.
- [x] Tüm backend testlerini dış socket bağlantıları engellenerek çalıştır
  (Windows asyncio için loopback serbest).
- [x] Bellekte bozulmuş patern/sinyal varyantlarını testlerin yakaladığını doğrula.
- [x] Python sözdizimi, CDSK yapısı ve diff kontrolünü çalıştır; sonuçları belgele.
- [x] BL-001/M1'i açık tut; sıradaki kontrollü tarayıcı testlerini handoff'a yaz.

## Uygulama kaydı

- Başlangıç: Git temiz, `6ed1a59`; devam talebi önceki beklemeyi kaldırdı.
- Test ortamı: Windows Python komut alias'ı çalışmadı; uygulamayla gelen
  Python 3.12 üzerinden ignore edilen `.venv` hazırlandı. Requirements değişmedi.
- Plan kapsamı test/belgedir; üretim kodu ve finansal davranış korunur.
- Task 1 tamamlandı: 14 test; Task 2 tamamlandı: 11 test, toplam 60 backend
  testi normal/dış ağ engelli koşuda geçti. Doji/FVG/RSI/çoğunluk için dört
  bellek içi mutasyon assertion ile yakalandı.
- Doğrulama sınırı: 39 Python dosyası ve manifestteki 39 belge/11 Boot bölümü
  kontrol edildi. Belgelenen upstream CDSK validator bu ortamda bulunamadı.
- Bağımsız inceleme: engelleyici bulgu yok; iki test modülü ayrıca çalıştırıldı.
  BL-001 içinde ertelenen düşük öncelikli kapsam: hammer/shooting-star kısa
  gölgesinde kayan noktasız tam eşitlik, yıldızlarda ilk gap'in tek başına
  bozulması, çift tepe/dipte %2 sınırına yakın eşit olmayan ekstremumlar.
