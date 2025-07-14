# BIST Teknik Analiz Uygulaması Konfigürasyonu

# En popüler BIST hisseleri
BIST_SYMBOLS = {
    "THYAO.IS": "Türk Hava Yolları",
    "TUPRS.IS": "Tüpraş",
    "BIMAS.IS": "BİM",
    "AKBNK.IS": "Akbank",
    "GARAN.IS": "Garanti Bankası",
    "ISCTR.IS": "İş Bankası",
    "HALKB.IS": "Halkbank",
    "VAKBN.IS": "Vakıfbank",
    "ARCLK.IS": "Arçelik",
    "KCHOL.IS": "Koç Holding",
    "EREGL.IS": "Ereğli Demir Çelik",
    "PETKM.IS": "Petkim",
    "TCELL.IS": "Turkcell",
    "ASELS.IS": "Aselsan",
    "TOASO.IS": "Tofaş",
    "SISE.IS": "Şişe Cam",
    "KOZAL.IS": "Koton",
    "MGROS.IS": "Migros",
    "FROTO.IS": "Ford Otosan",
    "SAHOL.IS": "Sabancı Holding",
    "DOHOL.IS": "Doğan Holding",
    "PGSUS.IS": "Pegasus",
    "EKGYO.IS": "Emlak Konut GYO",
    "VESTL.IS": "Vestel",
    "KOZAA.IS": "Koza Altın",
    "ENKAI.IS": "Enka İnşaat",
    "TAVHL.IS": "TAV Havalimanları",
    "ULKER.IS": "Ülker",
    "SOKM.IS": "Şok Marketler",
    "TATGD.IS": "TAT Gıda"
}

# Teknik indikatör konfigürasyonu
INDICATORS_CONFIG = {
    "ema_5": {
        "name": "EMA 5",
        "period": 5,
        "default": True,
        "color": "#ff6b6b"
    },
    "ema_8": {
        "name": "EMA 8", 
        "period": 8,
        "default": True,
        "color": "#4ecdc4"
    },
    "ema_13": {
        "name": "EMA 13",
        "period": 13,
        "default": True,
        "color": "#45b7d1"
    },
    "ema_21": {
        "name": "EMA 21",
        "period": 21,
        "default": True,
        "color": "#f9ca24"
    },
    "ema_50": {
        "name": "EMA 50",
        "period": 50,
        "default": True,
        "color": "#6c5ce7"
    },
    "ema_121": {
        "name": "EMA 121",
        "period": 121,
        "default": False,
        "color": "#a29bfe"
    },
    "rsi": {
        "name": "Göreceli Güç Endeksi (RSI)",
        "period": 14,
        "default": True,
        "overbought": 70,
        "oversold": 30
    },
    "macd": {
        "name": "MACD",
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "default": True
    },
    "bollinger": {
        "name": "Bollinger Bantları",
        "period": 20,
        "std": 2,
        "default": True
    },
    "stoch": {
        "name": "Stokastik Osilatör",
        "k_period": 14,
        "d_period": 3,
        "default": False
    },
    "williams_r": {
        "name": "Williams %R",
        "period": 14,
        "default": False
    },
    "cci": {
        "name": "Emtia Kanal Endeksi (CCI)",
        "period": 20,
        "default": False
    },
    "supertrend": {
        "name": "SuperTrend",
        "period": 10,
        "multiplier": 3.0,
        "default": True,
        "color_up": "#00ff88",
        "color_down": "#ff4757"
    },
    "ott": {
        "name": "OTT (Optimized Trend Tracker)",
        "period": 2,
        "percent": 1.4,
        "default": True,
        "color_up": "#2ed573",
        "color_down": "#ff3838"
    },
    "vwap": {
        "name": "VWAP",
        "period": 20,
        "default": True,
        "color": "#ff9ff3"
    }
}

# Alert konfigürasyonu
ALERT_CONFIG = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "volume_spike_multiplier": 2.0,
    "price_change_threshold": 5.0,  # Yüzde
    "update_interval": 300,  # Saniye (5 dakika)
}

# Grafik renkleri
CHART_COLORS = {
    "green": "#00ff00",
    "red": "#ff0000",
    "blue": "#0000ff",
    "orange": "#ffa500",
    "purple": "#800080",
    "gray": "#808080",
    "yellow": "#ffff00",
    "cyan": "#00ffff"
}

# Zaman aralıkları
TIME_PERIODS = {
    "1d": "1 Gün",
    "5d": "5 Gün", 
    "1mo": "1 Ay",
    "3mo": "3 Ay",
    "6mo": "6 Ay",
    "1y": "1 Yıl",
    "2y": "2 Yıl",
    "5y": "5 Yıl",
    "10y": "10 Yıl",
    "ytd": "Yıl Başından İtibaren",
    "max": "Maksimum"
} 