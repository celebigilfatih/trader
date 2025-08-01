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
    "TATGD.IS": "TAT Gıda",
    "DYOBY.IS": "D.Y.O Boya",
    "GMTAS.IS": "Gümüştaş",
    "YEOTK.IS": "Yeo Teknoloji",
    "BEGYO.IS": "Batı Eksen GYO",
    "FENER.IS": "Fenerbahçe",
    "METUR.IS": "Metin Tur",
    "ARDYZ.IS": "Ardemir Yatırım",
    "MERCN.IS": "Mercan Kimya",
    "OSMEN.IS": "Osmanlı Menkul",
    "TRILC.IS": "Tril Gayrimenkul",
    "MARBL.IS": "Tureks Madencilik",
    "KTLEV.IS": "Katılımevim Finansman",
    "MOPAS.IS": "Mopas Marketcilik",
    "TURSG.IS": "Türkiye Sigorta",
    "BIENY.IS": "Bien Yapı",
    "DOBUR.IS": "Doğuş Otomotiv",
    "KUYAS.IS": "Kuyumcukent",
    "DMRGD.IS": "Demirağ Gayrimenkul",
    "SAYAS.IS": "Sayas Yatırım",
    "GLCVY.IS": "Galata Çevre Yatırımları",
    "EGEPO.IS": "Ege Profil",
    "KONTR.IS": "Kontrolmatik Teknoloji",
    "ESEN.IS": "Esen Sistem",
    "NATEN.IS": "Naturel Yenilenebilir Enerji",
    "VAKFN.IS": "Vakıf Finansal Kiralama",
    "SEKFK.IS": "Şeker Finansal Kiralama",
    "GARFA.IS": "Garanti Faktoring",
    "ISFIN.IS": "İş Finansal Kiralama",
    "ULUFA.IS": "Ulu Faktoring",
    "LIDFA.IS": "Lider Faktoring",
    # BIST 100 Bankalar
    "YKBNK.IS": "Yapı Kredi Bankası",
    "ICBCT.IS": "ICBC Turkey Bank",
    "ALBRK.IS": "Albaraka Türk",
    "QNBFB.IS": "QNB Finansbank",
    "TSKB.IS": "Türkiye Sınai Kalkınma Bankası",
    # BIST 100 Gıda & İçecek
    "CCOLA.IS": "Coca Cola İçecek",
    "AEFES.IS": "Anadolu Efes",
    "KNFRT.IS": "Konfrut Gıda",
    "BANVT.IS": "Banvit",
    # BIST 100 Enerji
    "AKSEN.IS": "Aksa Enerji",
    "ZOREN.IS": "Zorlu Enerji",
    "AKENR.IS": "Akenerji",
    "ODAS.IS": "Odaş Elektrik",
    # BIST 100 Teknoloji
    "LOGO.IS": "Logo Yazılım",
    "ALCTL.IS": "Alcatel Lucent",
    "PENTA.IS": "Penta Teknoloji",
    "NETAS.IS": "Netaş Telekomünikasyon",
    # BIST 100 Holding
    "AGHOL.IS": "Anadolu Grubu Holding",
    "GUBRF.IS": "Gübre Fabrikaları",
    "PARSN.IS": "Parsan Makina",
    # BIST 100 Perakende
    "BIZIM.IS": "Bizim Toptan",
    "DOAS.IS": "Doğaş Otomotiv",
    "SELEC.IS": "Selçuk Ecza Deposu",
    # BIST 100 İnşaat & Çimento
    "EDIP.IS": "Edip Gayrimenkul",
    "ENJSA.IS": "Enerjisa Enerji",
    "CEMTS.IS": "Çemtaş Çelik",
    "KRDMD.IS": "Kardemir",
    "AKGRT.IS": "Aksigorta",
    # BIST 100 Otomotiv
    "OTKAR.IS": "Otokar",
    "TTRAK.IS": "Türk Traktör",
    "EGEEN.IS": "Ege Endüstri",
    # BIST 100 Tekstil
    "BRSAN.IS": "Borusan Mannesmann",
    "DMSAS.IS": "Demisaş Döküm",
    "ISGYO.IS": "İş GYO",
    # BIST 100 Sağlık
    "DEVA.IS": "Deva Holding",
    "ECILC.IS": "Eczacıbaşı İlaç",
    "LKMNH.IS": "Lokman Hekim",
    # BIST 100 Turizm
    "MAALT.IS": "Marmaris Altın Tesisleri",
    "AYDEM.IS": "Aydem Enerji",
    # BIST 100 Diğer
    "SODA.IS": "Soda Sanayii",
    "CANTE.IS": "Çanakkale Seramik",
    "KERVT.IS": "Kervan Gıda",
    "BAGFS.IS": "Bagfaş",
    "TTKOM.IS": "Türk Telekom",
    "GLYHO.IS": "Global Yatırım Holding",
    "IHEVA.IS": "İheva İnşaat",
    "KLMSN.IS": "Klimasan Klima",
    "MAVI.IS": "Mavi Giyim"
}

# Teknik indikatör konfigürasyonu
INDICATORS_CONFIG = {
    "ema_5": {
        "name": "EMA 5",
        "period": 5,
        "default": False,
        "color": "#ff6b6b"
    },
    "ema_8": {
        "name": "EMA 8", 
        "period": 8,
        "default": False,
        "color": "#4ecdc4"
    },
    "ema_13": {
        "name": "EMA 13",
        "period": 13,
        "default": False,
        "color": "#45b7d1"
    },
    "ema_21": {
        "name": "EMA 21",
        "period": 21,
        "default": False,
        "color": "#f9ca24"
    },
    "ema_50": {
        "name": "EMA 50",
        "period": 50,
        "default": False,
        "color": "#6c5ce7"
    },
    "ema_121": {
        "name": "EMA 121",
        "period": 121,
        "default": False,
        "color": "#a29bfe"
    },
    "ma_200": {
        "name": "MA 200",
        "period": 200,
        "default": False,
        "color": "#e74c3c"
    },
    "vwma_5": {
        "name": "VWMA 5",
        "period": 5,
        "default": False,
        "color": "#9b59b6"
    },
    "vwema_5": {
        "name": "VWEMA 5",
        "period": 5,
        "default": True,
        "color": "#e67e22"
    },
    "vwema_20": {
        "name": "VWEMA 20",
        "period": 20,
        "default": True,
        "color": "#3498db"
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
        "default": False
    },
    "bollinger": {
        "name": "Bollinger Bantları",
        "period": 20,
        "std": 2,
        "default": False
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
        "default": False,
        "color_up": "#00ff88",
        "color_down": "#ff4757"
    },
    "ott": {
        "name": "OTT (Optimized Trend Tracker)",
        "period": 2,
        "percent": 1.4,
        "default": False,
        "color_up": "#2ed573",
        "color_down": "#ff3838"
    },
    "vwap": {
        "name": "VWAP",
        "period": 20,
        "default": True,
        "color": "#ff9ff3"
    },
    # Yeni gelişmiş indikatörler
    "fvg": {
        "name": "Fair Value Gap (FVG)",
        "threshold_percent": 0.2,
        "default": False,
        "color_bullish": "#00b894",
        "color_bearish": "#d63031"
    },
    "order_block": {
        "name": "Order Block",
        "lookback": 20,
        "threshold_percent": 0.5,
        "default": False,
        "color_bullish": "#0984e3",
        "color_bearish": "#e84393"
    },
    "bos": {
        "name": "Break of Structure (BOS)",
        "lookback": 50,
        "swing_threshold": 0.5,
        "default": False,
        "color_bullish": "#00cec9",
        "color_bearish": "#fd79a8"
    },
    "fvg_ob_combo": {
        "name": "FVG + Order Block Kombosu",
        "default": False,
        "color_bullish": "#6c5ce7",
        "color_bearish": "#e17055"
    },
    "fvg_bos_combo": {
        "name": "FVG + Break of Structure Kombosu",
        "default": False,
        "color_bullish": "#00b894",
        "color_bearish": "#d63031"
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