import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class AdvancedPatternRecognition:
    """Gelişmiş fiyat yapısı analizi ve desen tanıma sistemi"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.patterns = {}
        self.structure_points = {}
        self.fvg_zones = {}
        self.order_blocks = {}
    
    def detect_fair_value_gaps(self, threshold_percent: float = 0.2) -> Dict[str, pd.DataFrame]:
        """
        Fair Value Gap (FVG) tespiti yapar
        
        FVG, fiyatın hızlı hareket ettiği ve doldurmadığı boşluklardır.
        Bullish FVG: Düşük fiyat, bir önceki mumun yüksek fiyatından daha yüksek
        Bearish FVG: Yüksek fiyat, bir önceki mumun düşük fiyatından daha düşük
        
        Args:
            threshold_percent: Minimum boşluk yüzdesi (fiyatın yüzdesi olarak)
            
        Returns:
            Dict: Bullish ve bearish FVG'leri içeren DataFrame'ler
        """
        # Boş DataFrame'ler oluştur
        bullish_fvg = pd.DataFrame(columns=['date', 'low', 'high', 'size', 'filled'])
        bearish_fvg = pd.DataFrame(columns=['date', 'low', 'high', 'size', 'filled'])
        
        # FVG tespiti için minimum 3 mum gerekli
        if len(self.data) < 3:
            return {'bullish': bullish_fvg, 'bearish': bearish_fvg}
        
        # Bullish FVG tespiti
        for i in range(2, len(self.data)):
            # Mum 1 ve Mum 3 arasında boşluk var mı?
            if self.data['Low'].iloc[i] > self.data['High'].iloc[i-2]:
                gap_size = self.data['Low'].iloc[i] - self.data['High'].iloc[i-2]
                threshold = self.data['Close'].iloc[i-2] * threshold_percent / 100
                
                # Boşluk yeterince büyük mü?
                if gap_size > threshold:
                    # FVG dolduruldu mu kontrol et
                    filled = False
                    for j in range(i+1, len(self.data)):
                        if self.data['Low'].iloc[j] <= self.data['High'].iloc[i-2]:
                            filled = True
                            break
                    
                    # FVG'yi kaydet
                    bullish_fvg = pd.concat([bullish_fvg, pd.DataFrame({
                        'date': [self.data.index[i]],
                        'low': [self.data['High'].iloc[i-2]],
                        'high': [self.data['Low'].iloc[i]],
                        'size': [gap_size],
                        'filled': [filled]
                    })], ignore_index=True)
        
        # Bearish FVG tespiti
        for i in range(2, len(self.data)):
            # Mum 1 ve Mum 3 arasında boşluk var mı?
            if self.data['High'].iloc[i] < self.data['Low'].iloc[i-2]:
                gap_size = self.data['Low'].iloc[i-2] - self.data['High'].iloc[i]
                threshold = self.data['Close'].iloc[i-2] * threshold_percent / 100
                
                # Boşluk yeterince büyük mü?
                if gap_size > threshold:
                    # FVG dolduruldu mu kontrol et
                    filled = False
                    for j in range(i+1, len(self.data)):
                        if self.data['High'].iloc[j] >= self.data['Low'].iloc[i-2]:
                            filled = True
                            break
                    
                    # FVG'yi kaydet
                    bearish_fvg = pd.concat([bearish_fvg, pd.DataFrame({
                        'date': [self.data.index[i]],
                        'low': [self.data['High'].iloc[i]],
                        'high': [self.data['Low'].iloc[i-2]],
                        'size': [gap_size],
                        'filled': [filled]
                    })], ignore_index=True)
        
        self.fvg_zones = {'bullish': bullish_fvg, 'bearish': bearish_fvg}
        return self.fvg_zones
    
    def detect_order_blocks(self, lookback: int = 20, threshold_percent: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        Order Block tespiti yapar
        
        Order Block, fiyatın yön değiştirmeden önceki son momentum mumudur.
        Bullish OB: Düşüş trendinde son düşüş mumu, ardından yükseliş gelir
        Bearish OB: Yükseliş trendinde son yükseliş mumu, ardından düşüş gelir
        
        Args:
            lookback: Geriye dönük bakılacak mum sayısı
            threshold_percent: Minimum fiyat hareketi yüzdesi
            
        Returns:
            Dict: Bullish ve bearish Order Block'ları içeren DataFrame'ler
        """
        # Boş DataFrame'ler oluştur
        bullish_ob = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'strength'])
        bearish_ob = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'strength'])
        
        # Minimum veri kontrolü
        if len(self.data) < lookback + 3:
            return {'bullish': bullish_ob, 'bearish': bearish_ob}
        
        # Son lookback kadar mumu analiz et
        for i in range(3, min(len(self.data), lookback + 3)):
            # Bullish Order Block tespiti
            # Düşüş mumu + ardından yükseliş
            if (self.data['Close'].iloc[-i] < self.data['Open'].iloc[-i] and  # Düşüş mumu
                self.data['Close'].iloc[-i+1] > self.data['Open'].iloc[-i+1] and  # Yükseliş mumu
                self.data['Close'].iloc[-i+2] > self.data['Close'].iloc[-i+1]):  # Devam eden yükseliş
                
                # Fiyat hareketi yeterince büyük mü?
                price_move = (self.data['Close'].iloc[-i+2] - self.data['Low'].iloc[-i]) / self.data['Low'].iloc[-i] * 100
                if price_move > threshold_percent:
                    # Order Block gücünü hesapla (fiyat hareketi büyüklüğüne göre)
                    strength = min(100, price_move * 10)
                    
                    # Order Block'u kaydet
                    bullish_ob = pd.concat([bullish_ob, pd.DataFrame({
                        'date': [self.data.index[-i]],
                        'open': [self.data['Open'].iloc[-i]],
                        'high': [self.data['High'].iloc[-i]],
                        'low': [self.data['Low'].iloc[-i]],
                        'close': [self.data['Close'].iloc[-i]],
                        'strength': [strength]
                    })], ignore_index=True)
            
            # Bearish Order Block tespiti
            # Yükseliş mumu + ardından düşüş
            if (self.data['Close'].iloc[-i] > self.data['Open'].iloc[-i] and  # Yükseliş mumu
                self.data['Close'].iloc[-i+1] < self.data['Open'].iloc[-i+1] and  # Düşüş mumu
                self.data['Close'].iloc[-i+2] < self.data['Close'].iloc[-i+1]):  # Devam eden düşüş
                
                # Fiyat hareketi yeterince büyük mü?
                price_move = (self.data['High'].iloc[-i] - self.data['Close'].iloc[-i+2]) / self.data['High'].iloc[-i] * 100
                if price_move > threshold_percent:
                    # Order Block gücünü hesapla (fiyat hareketi büyüklüğüne göre)
                    strength = min(100, price_move * 10)
                    
                    # Order Block'u kaydet
                    bearish_ob = pd.concat([bearish_ob, pd.DataFrame({
                        'date': [self.data.index[-i]],
                        'open': [self.data['Open'].iloc[-i]],
                        'high': [self.data['High'].iloc[-i]],
                        'low': [self.data['Low'].iloc[-i]],
                        'close': [self.data['Close'].iloc[-i]],
                        'strength': [strength]
                    })], ignore_index=True)
        
        self.order_blocks = {'bullish': bullish_ob, 'bearish': bearish_ob}
        return self.order_blocks
    
    def detect_break_of_structure(self, lookback: int = 50, swing_threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
        """
        Break of Structure (BOS) tespiti yapar
        
        BOS, fiyatın önceki swing high/low noktalarını kırmasıdır.
        Bullish BOS: Fiyat önceki yüksek noktayı kırar
        Bearish BOS: Fiyat önceki düşük noktayı kırar
        
        Args:
            lookback: Geriye dönük bakılacak mum sayısı
            swing_threshold: Swing noktası olarak kabul edilecek minimum fiyat hareketi yüzdesi
            
        Returns:
            Dict: Bullish ve bearish BOS'ları içeren DataFrame'ler
        """
        # Önce swing high/low noktalarını tespit et
        self._identify_structure_points(lookback, swing_threshold)
        
        # Boş DataFrame'ler oluştur
        bullish_bos = pd.DataFrame(columns=['date', 'price', 'prev_high', 'strength', 'volume_confirmation'])
        bearish_bos = pd.DataFrame(columns=['date', 'price', 'prev_low', 'strength', 'volume_confirmation'])
        
        # Minimum veri kontrolü
        if len(self.data) < 10 or not self.structure_points:
            return {'bullish': bullish_bos, 'bearish': bearish_bos}
        
        # Swing high/low noktalarını kullanarak BOS tespit et
        highs = self.structure_points['highs']
        lows = self.structure_points['lows']
        
        # Son lookback kadar mumu analiz et
        for i in range(10, min(len(self.data), lookback)):
            current_idx = len(self.data) - i
            current_price = self.data['Close'].iloc[current_idx]
            current_date = self.data.index[current_idx]
            current_volume = self.data['Volume'].iloc[current_idx]
            
            # Hacim onayı için ortalama hacmi hesapla
            avg_volume = self.data['Volume'].iloc[max(0, current_idx-20):current_idx].mean()
            volume_confirmation = current_volume > avg_volume * 1.2  # %20 fazla hacim
            
            # Bullish BOS: Önceki swing high kırıldı mı?
            prev_highs = [h for h in highs if h['idx'] < current_idx]
            if prev_highs:
                # En yakın önceki swing high'ı bul
                recent_high = max([h for h in prev_highs], key=lambda x: x['price'])
                
                # BOS koşulu: Fiyat önceki swing high'ı kırmalı ve kapanış da üstünde olmalı
                if (current_price > recent_high['price'] and 
                    self.data['High'].iloc[current_idx] > recent_high['price']):
                    
                    # BOS gücünü hesapla (kırılan yapının büyüklüğüne göre)
                    price_break_percent = (current_price - recent_high['price']) / recent_high['price'] * 100
                    strength = min(100, price_break_percent * 15 + recent_high['strength'] * 0.3)
                    
                    # Hacim onayı varsa güçü artır
                    if volume_confirmation:
                        strength *= 1.3
                    
                    # BOS'u kaydet
                    bullish_bos = pd.concat([bullish_bos, pd.DataFrame({
                        'date': [current_date],
                        'price': [current_price],
                        'prev_high': [recent_high['price']],
                        'strength': [strength],
                        'volume_confirmation': [volume_confirmation]
                    })], ignore_index=True)
            
            # Bearish BOS: Önceki swing low kırıldı mı?
            prev_lows = [l for l in lows if l['idx'] < current_idx]
            if prev_lows:
                # En yakın önceki swing low'u bul
                recent_low = min([l for l in prev_lows], key=lambda x: x['price'])
                
                # BOS koşulu: Fiyat önceki swing low'u kırmalı ve kapanış da altında olmalı
                if (current_price < recent_low['price'] and 
                    self.data['Low'].iloc[current_idx] < recent_low['price']):
                    
                    # BOS gücünü hesapla (kırılan yapının büyüklüğüne göre)
                    price_break_percent = (recent_low['price'] - current_price) / recent_low['price'] * 100
                    strength = min(100, price_break_percent * 15 + recent_low['strength'] * 0.3)
                    
                    # Hacim onayı varsa güçü artır
                    if volume_confirmation:
                        strength *= 1.3
                    
                    # BOS'u kaydet
                    bearish_bos = pd.concat([bearish_bos, pd.DataFrame({
                        'date': [current_date],
                        'price': [current_price],
                        'prev_low': [recent_low['price']],
                        'strength': [strength],
                        'volume_confirmation': [volume_confirmation]
                    })], ignore_index=True)
        
        return {'bullish': bullish_bos, 'bearish': bearish_bos}
    
    def _identify_structure_points(self, lookback: int = 50, swing_threshold: float = 0.5):
        """
        Swing high/low noktalarını tespit eder
        
        Args:
            lookback: Geriye dönük bakılacak mum sayısı
            swing_threshold: Swing noktası olarak kabul edilecek minimum fiyat hareketi yüzdesi
        """
        if len(self.data) < 10:
            self.structure_points = {'highs': [], 'lows': []}
            return
        
        highs = []
        lows = []
        
        # Daha hassas swing tespit parametreleri
        min_swing_distance = 5  # Swing noktaları arası minimum mesafe
        strength_multiplier = 2.0  # Güç hesaplama çarpanı
        
        # Son lookback kadar mumu analiz et
        for i in range(5, min(len(self.data) - 5, lookback)):
            current_idx = len(self.data) - i
            current_high = self.data['High'].iloc[current_idx]
            current_low = self.data['Low'].iloc[current_idx]
            current_volume = self.data['Volume'].iloc[current_idx]
            
            # Hacim onayı için ortalama hacmi hesapla
            avg_volume = self.data['Volume'].iloc[max(0, current_idx-10):current_idx+10].mean()
            volume_strength = min(2.0, current_volume / avg_volume) if avg_volume > 0 else 1.0
            
            # Swing High kontrolü - önceki ve sonraki 3 mumla karşılaştır
            is_swing_high = True
            for j in range(1, 4):
                if (current_idx - j >= 0 and current_high <= self.data['High'].iloc[current_idx - j]) or \
                   (current_idx + j < len(self.data) and current_high <= self.data['High'].iloc[current_idx + j]):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                # Swing high gücünü hesapla
                price_range = max(self.data['High'].iloc[max(0, current_idx-10):current_idx+10]) - \
                             min(self.data['Low'].iloc[max(0, current_idx-10):current_idx+10])
                
                if price_range > 0:
                    # Çevredeki mumlarla fark yüzdesi
                    nearby_highs = [self.data['High'].iloc[current_idx + k] for k in range(-3, 4) 
                                   if 0 <= current_idx + k < len(self.data) and k != 0]
                    if nearby_highs:
                        avg_nearby_high = sum(nearby_highs) / len(nearby_highs)
                        height_advantage = (current_high - avg_nearby_high) / avg_nearby_high * 100
                        
                        # Güç hesaplama: yükseklik avantajı + hacim onayı
                        strength = min(100, height_advantage * strength_multiplier * volume_strength)
                        
                        # Minimum güç eşiği kontrolü
                        if strength >= swing_threshold * 10:  # swing_threshold'u güç için uyarla
                            # Çok yakın swing high var mı kontrol et
                            too_close = any(abs(h['idx'] - current_idx) < min_swing_distance for h in highs)
                            if not too_close:
                                highs.append({
                                    'idx': current_idx,
                                    'price': current_high,
                                    'strength': strength,
                                    'volume_strength': volume_strength,
                                    'date': self.data.index[current_idx]
                                })
            
            # Swing Low kontrolü - önceki ve sonraki 3 mumla karşılaştır
            is_swing_low = True
            for j in range(1, 4):
                if (current_idx - j >= 0 and current_low >= self.data['Low'].iloc[current_idx - j]) or \
                   (current_idx + j < len(self.data) and current_low >= self.data['Low'].iloc[current_idx + j]):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                # Swing low gücünü hesapla
                price_range = max(self.data['High'].iloc[max(0, current_idx-10):current_idx+10]) - \
                             min(self.data['Low'].iloc[max(0, current_idx-10):current_idx+10])
                
                if price_range > 0:
                    # Çevredeki mumlarla fark yüzdesi
                    nearby_lows = [self.data['Low'].iloc[current_idx + k] for k in range(-3, 4) 
                                  if 0 <= current_idx + k < len(self.data) and k != 0]
                    if nearby_lows:
                        avg_nearby_low = sum(nearby_lows) / len(nearby_lows)
                        depth_advantage = (avg_nearby_low - current_low) / avg_nearby_low * 100
                        
                        # Güç hesaplama: derinlik avantajı + hacim onayı
                        strength = min(100, depth_advantage * strength_multiplier * volume_strength)
                        
                        # Minimum güç eşiği kontrolü
                        if strength >= swing_threshold * 10:  # swing_threshold'u güç için uyarla
                            # Çok yakın swing low var mı kontrol et
                            too_close = any(abs(l['idx'] - current_idx) < min_swing_distance for l in lows)
                            if not too_close:
                                lows.append({
                                    'idx': current_idx,
                                    'price': current_low,
                                    'strength': strength,
                                    'volume_strength': volume_strength,
                                    'date': self.data.index[current_idx]
                                })
        
        # Güce göre sırala ve en güçlü olanları seç
        highs = sorted(highs, key=lambda x: x['strength'], reverse=True)[:20]
        lows = sorted(lows, key=lambda x: x['strength'], reverse=True)[:20]
        
        self.structure_points = {'highs': highs, 'lows': lows}
    
    def get_fvg_order_block_combo(self) -> List[Dict]:
        """
        FVG ve Order Block kombinasyonlarını tespit eder
        Bu kombinasyonlar güçlü alım/satım bölgeleridir
        
        Returns:
            List: FVG ve Order Block kombinasyonları
        """
        # Eğer henüz tespit edilmediyse, FVG ve OB'leri tespit et
        if not self.fvg_zones:
            self.detect_fair_value_gaps()
        if not self.order_blocks:
            self.detect_order_blocks()
        
        combos = []
        
        # Bullish FVG + Bullish OB kombinasyonları
        for _, fvg in self.fvg_zones['bullish'].iterrows():
            for _, ob in self.order_blocks['bullish'].iterrows():
                # FVG ve OB yakın mı?
                if abs((fvg['low'] - ob['high']) / ob['high']) < 0.02:  # %2 içinde
                    combos.append({
                        'type': 'bullish',
                        'date': fvg['date'],
                        'fvg_zone': (fvg['low'], fvg['high']),
                        'order_block': (ob['low'], ob['high']),
                        'strength': (fvg['size'] + ob['strength']) / 2  # Ortalama güç
                    })
        
        # Bearish FVG + Bearish OB kombinasyonları
        for _, fvg in self.fvg_zones['bearish'].iterrows():
            for _, ob in self.order_blocks['bearish'].iterrows():
                # FVG ve OB yakın mı?
                if abs((fvg['high'] - ob['low']) / ob['low']) < 0.02:  # %2 içinde
                    combos.append({
                        'type': 'bearish',
                        'date': fvg['date'],
                        'fvg_zone': (fvg['low'], fvg['high']),
                        'order_block': (ob['low'], ob['high']),
                        'strength': (fvg['size'] + ob['strength']) / 2  # Ortalama güç
                    })
        
        return combos
    
    def get_fvg_bos_combo(self) -> List[Dict]:
        """
        FVG ve Break of Structure kombinasyonlarını tespit eder
        Bu kombinasyonlar trend dönüş noktalarını işaret eder
        
        Returns:
            List: FVG ve BOS kombinasyonları
        """
        from .config import INDICATORS_CONFIG
        
        # Konfigürasyondan parametreleri al
        config = INDICATORS_CONFIG.get('fvg_bos_combo', {})
        time_window = config.get('time_window', 3)
        strength_threshold = config.get('strength_threshold', 30)
        proximity_threshold = config.get('proximity_threshold', 0.015)
        
        # Eğer henüz tespit edilmediyse, FVG ve BOS'ları tespit et
        if not self.fvg_zones:
            self.detect_fair_value_gaps()
        
        bos_data = self.detect_break_of_structure()
        
        combos = []
        
        # Bullish FVG + Bullish BOS kombinasyonları
        for _, fvg in self.fvg_zones['bullish'].iterrows():
            for _, bos in bos_data['bullish'].iterrows():
                # Zaman yakınlığı kontrolü (ayarlanabilir gün farkı)
                time_diff = abs((fvg['date'] - bos['date']).days)
                if time_diff <= time_window:
                    # Fiyat yakınlığı kontrolü
                    fvg_mid = (fvg['low'] + fvg['high']) / 2
                    price_diff = abs(fvg_mid - bos['price']) / bos['price']
                    
                    # Güç kontrolü
                    combo_strength = (fvg['size'] + bos['strength']) / 2
                    
                    if price_diff <= proximity_threshold and combo_strength >= strength_threshold:
                        combos.append({
                            'type': 'bullish',
                            'date': max(fvg['date'], bos['date']),
                            'fvg_zone': (fvg['low'], fvg['high']),
                            'bos_price': bos['price'],
                            'prev_high': bos['prev_high'],
                            'strength': combo_strength,
                            'time_diff': time_diff,
                            'price_proximity': price_diff,
                            'confidence': min(100, combo_strength * (1 - price_diff) * (1 - time_diff/time_window))
                        })
        
        # Bearish FVG + Bearish BOS kombinasyonları
        for _, fvg in self.fvg_zones['bearish'].iterrows():
            for _, bos in bos_data['bearish'].iterrows():
                # Zaman yakınlığı kontrolü (ayarlanabilir gün farkı)
                time_diff = abs((fvg['date'] - bos['date']).days)
                if time_diff <= time_window:
                    # Fiyat yakınlığı kontrolü
                    fvg_mid = (fvg['low'] + fvg['high']) / 2
                    price_diff = abs(fvg_mid - bos['price']) / bos['price']
                    
                    # Güç kontrolü
                    combo_strength = (fvg['size'] + bos['strength']) / 2
                    
                    if price_diff <= proximity_threshold and combo_strength >= strength_threshold:
                        combos.append({
                            'type': 'bearish',
                            'date': max(fvg['date'], bos['date']),
                            'fvg_zone': (fvg['low'], fvg['high']),
                            'bos_price': bos['price'],
                            'prev_low': bos['prev_low'],
                            'strength': combo_strength,
                            'time_diff': time_diff,
                            'price_proximity': price_diff,
                            'confidence': min(100, combo_strength * (1 - price_diff) * (1 - time_diff/time_window))
                        })
        
        # Güce göre sırala (en güçlü kombinasyonlar önce)
        combos.sort(key=lambda x: x['confidence'], reverse=True)
        
        return combos
    
    def get_latest_signals(self) -> Dict[str, List]:
        """
        Son tespit edilen tüm sinyalleri döndürür
        
        Returns:
            Dict: Tüm sinyal türleri
        """
        # Tüm analizleri yap
        self.detect_fair_value_gaps()
        self.detect_order_blocks()
        bos_data = self.detect_break_of_structure()
        fvg_ob_combos = self.get_fvg_order_block_combo()
        fvg_bos_combos = self.get_fvg_bos_combo()
        
        # Son 10 günde doldurulmamış FVG'leri filtrele
        recent_bullish_fvg = self.fvg_zones['bullish'][
            (self.fvg_zones['bullish']['filled'] == False) & 
            (self.fvg_zones['bullish']['date'] >= self.data.index[-10])]
        
        recent_bearish_fvg = self.fvg_zones['bearish'][
            (self.fvg_zones['bearish']['filled'] == False) & 
            (self.fvg_zones['bearish']['date'] >= self.data.index[-10])]
        
        # Son 10 günde oluşan Order Block'ları filtrele
        recent_bullish_ob = self.order_blocks['bullish'][
            self.order_blocks['bullish']['date'] >= self.data.index[-10]]
        
        recent_bearish_ob = self.order_blocks['bearish'][
            self.order_blocks['bearish']['date'] >= self.data.index[-10]]
        
        return {
            'fvg': {
                'bullish': recent_bullish_fvg.to_dict('records'),
                'bearish': recent_bearish_fvg.to_dict('records')
            },
            'order_blocks': {
                'bullish': recent_bullish_ob.to_dict('records'),
                'bearish': recent_bearish_ob.to_dict('records')
            },
            'break_of_structure': {
                'bullish': bos_data['bullish'].to_dict('records'),
                'bearish': bos_data['bearish'].to_dict('records')
            },
            'fvg_ob_combo': fvg_ob_combos,
            'fvg_bos_combo': fvg_bos_combos
        }