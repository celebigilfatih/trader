# Bounded Contexts

## Current Contexts (Legacy)

| Context / module | Responsibility | Owns data | Does not own |
|---|---|---|---|
| Streamlit UI (`app.py`) | Navigasyon, kullanıcı girdisi, görünüm, grafik ve orchestration | Session state ve geçici UI seçimleri | Piyasa verisinin doğruluğu, indikatör formülleri |
| Market Data (`data_fetcher.py`) | Yahoo Finance verisini OHLCV sözleşmesine dönüştürmek | Çalışma anındaki normalize DataFrame | Sinyal, UI, kalıcı portföy |
| Technical Analysis (`technical_analysis.py`, pattern modülleri) | İndikatör ve formasyon hesaplamak | Türetilmiş gösterge serileri | Veri çekme, kullanıcı görünümü |
| Signals & Alerts (`alert_system.py`) | Sinyal/risk/uyarı üretmek ve isteğe bağlı bildirmek | Geçici alarm çıktıları; çağrılırsa alarm CSV'si | Piyasa veri kaynağı, portföy muhasebesi |
| Screening (`stock_screener.py`, `day_trader.py`) | Sembol listesini kurallarla taramak | Geçici tarama sonuçları ve aktif day-trader nesne durumu | Emir yürütme, kalıcı pozisyon defteri |
| Experimental ML (`simple_ml_predictor.py`, `ml_predictor.py`) | Özellik üretmek, yerel model eğitmek ve tahmin yapmak | Bellekte model/scaler; isteğe bağlı model dosyası | Finansal garanti, veri kaynağı |
| News & Sentiment (`sentiment_analyzer.py`) | RSS toplamak, sembol ve duygu skoru çıkarmak | Bellek içi günlük cache ve mock tablolar | Haber doğruluğu, piyasa emri |
| Configuration (`config.py`) | Sembol ve gösterge/uyarı parametrelerini tanımlamak | Statik Python sabitleri | Kullanıcıya özel ayar saklama |

## Target Contexts (Accepted Plan)

| Context / module | Responsibility | Owns data | Does not own |
|---|---|---|---|
| Gateway UI (`Next.js`) | Günlük karar cockpit’i, watchlist, portföy ve alarm paneli | Oturumlu kullanıcı görünümleri, filtre durumu | Hesaplama motoru, veri çekim stratejisi |
| API Core (`FastAPI`) | Tüm analiz, tarama, alarm ve keşif akışlarını API sözleşmesiyle sunmak | Kalıcı domain tabloları, iş sonuçları | UI kurgusu ve sunum formatı |
| Auth service | Davet + magic link doğrulama | Token, oturum, davet, cihaz-oturum durumu | Analiz ve risk hesapları |
| Market Data service | Provider bazlı OHLCV ve şirket verisi standardizasyonu | OHLCV snapshot’ları, şirket metadata | Alarm karar mantığı, portföy muhasebesi |
| Portfolio service | Paper portföyün sadece sembol, adet, ortalama maliyet akışı | Holdings durumu ve snapshot P&L hesapları | Emir verme, nakit/komisyon yönetimi |
| Screener service | Kural tabanlı keşif ve scan sonuçları | Screener çalıştırmaları ve sonuç geçmişi | Piyasa verisi temini |
| Notification service | In-app + e-posta alarm kuyruğu | Bildirim outbox ve durum takibi | Model eğitimi, fiyat öngörüsü |
| Operations service | Job scheduling, retry, health, observability | Job durumu, sağlık metrikleri | İş mantığı (sinyal/analiz)
| AI lab | Deneysel ML/NLP tahminlerini üretmek, versiyonlamak | Tahmin denemeleri, model metadata | Core karar puanı, core alarm kararı |
