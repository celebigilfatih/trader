import pandas as pd
import requests
from typing import Dict, List, Optional
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from math import isfinite
import feedparser
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    """Haber ve sosyal medya duygu analizi"""
    
    def __init__(self):
        self.news_sources = {
            'investing_rss': 'https://tr.investing.com/rss/news.rss',
            'finans_rss': 'https://www.finansgundem.com/rss.xml',
            'bloomberg_rss': 'https://www.bloomberg.com.tr/rss',
            'aa_finans': 'https://www.aa.com.tr/tr/rss/default?cat=ekonomi',
            'hurriyet_ekonomi': 'https://www.hurriyet.com.tr/rss/ekonomi'
        }
        self.sentiment_cache = {}
        self.last_fetch_diagnostics = []
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.symbol_sentiments = {
            'THYAO': 0.2,   # Türk Hava Yolları - nötr/pozitif
            'AKBNK': 0.1,   # Akbank - hafif pozitif
            'GARAN': 0.15,  # Garanti - hafif pozitif
            'TUPRS': 0.3,   # Tüpraş - pozitif
            'BIMAS': 0.05,  # BİM - nötr
            'TKFEN': 0.25,  # Tekfen - pozitif
            'SAHOL': 0.1,   # Sabancı Holding - hafif pozitif
            'ISCTR': 0.0,   # İş Bankası - nötr
            'ASELS': 0.2,   # Aselsan - pozitif
            'TOASO': 0.15,  # Tofaş - hafif pozitif
            'USAK': 0.1,    # Uşak Seramik
            'OZSUB': 0.05,  # Özsu Boya
            'YIGIT': 0.15,  # Yiğit Yatırım
            'ACSEL': 0.1,   # Acıselsan
            'OBAMS': 0.05,  # Obams
            'GSRAY': 0.2,   # Galatasaray
            'DMRDG': 0.1,   # Demirdöküm
            'TEHOL': 0.15   # Teknosa
        }
    
    def get_basic_sentiment_score(self, symbol: str) -> Dict:
        """Basit sentiment skoru hesaplar"""
        
        # Önce cache kontrol et
        cache_key = f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        try:
            # Basit sentiment hesaplama (gerçek uygulamada API'ler kullanılır)
            sentiment_score = self._calculate_mock_sentiment(symbol)
            
            result = {
                'symbol': symbol,
                'sentiment_score': sentiment_score,
                'sentiment_label': self._get_sentiment_label(sentiment_score),
                'confidence': 0.7,  # Mock confidence
                'news_count': 5,    # Mock news count
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Cache'e kaydet
            self.sentiment_cache[cache_key] = result
            return result
            
        except Exception as e:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'sentiment_label': 'Neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _calculate_mock_sentiment(self, symbol: str) -> float:
        """Mock sentiment hesaplama (demo amaçlı)"""
        
        # Symbol'e göre basit sentiment skoru
        # Gerçek uygulamada NLP API'leri kullanılır
        
        base_sentiment = self.symbol_sentiments.get(symbol, 0.0)
        
        # Rastgele küçük değişiklik ekle (piyasa koşullarını simüle eder)
        import random
        random.seed(int(time.time()) // 3600)  # Saatlik değişim
        noise = random.uniform(-0.1, 0.1)
        
        final_sentiment = max(-1.0, min(1.0, base_sentiment + noise))
        return round(final_sentiment, 2)
    
    def _get_sentiment_label(self, score: float) -> str:
        """Sentiment skorunu etikete dönüştürür"""
        if score > 0.2:
            return 'Pozitif'
        elif score < -0.2:
            return 'Negatif'
        else:
            return 'Nötr'
    
    def get_market_sentiment(self, symbols: List[str]) -> Dict:
        """Piyasa geneli sentiment analizi"""
        
        sentiments = []
        for symbol in symbols[:10]:  # İlk 10 hisse için
            sentiment = self.get_basic_sentiment_score(symbol)
            if 'sentiment_score' in sentiment:
                sentiments.append(sentiment['sentiment_score'])
        
        if not sentiments:
            return {
                'market_sentiment': 0.0,
                'market_label': 'Nötr',
                'positive_stocks': 0,
                'negative_stocks': 0,
                'neutral_stocks': 0
            }
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        positive_count = len([s for s in sentiments if s > 0.1])
        negative_count = len([s for s in sentiments if s < -0.1])
        neutral_count = len(sentiments) - positive_count - negative_count
        
        return {
            'market_sentiment': round(avg_sentiment, 2),
            'market_label': self._get_sentiment_label(avg_sentiment),
            'positive_stocks': positive_count,
            'negative_stocks': negative_count,
            'neutral_stocks': neutral_count,
            'total_analyzed': len(sentiments)
        }
    
    def get_news_headlines(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Mock haber başlıkları (demo amaçlı)"""
        
        mock_headlines = {
            'THYAO': [
                'THY 2024 yılında güçlü büyüme hedefliyor',
                'Türk Hava Yolları yeni destinasyonlar açıyor',
                'THY kargo işletmesinde rekor büyüme',
                'Analistler THY için yükseliş bekliyor',
                'THY filosunu genişletmeye devam ediyor'
            ],
            'AKBNK': [
                'Akbank dijital bankacılıkta öncü olmaya devam ediyor',
                'AKBNK kredi portföyünü güçlendiriyor',
                'Akbank temettü ödemesi açıklandı',
                'Teknoloji yatırımları Akbank\'ı öne çıkarıyor',
                'AKBNK 2024 karlılık hedeflerini açıkladı'
            ]
        }
        
        headlines = mock_headlines.get(symbol, [
            f'{symbol} ile ilgili güncel gelişmeler',
            f'{symbol} analist raporları olumlu',
            f'{symbol} şirketi büyüme stratejisini açıkladı',
            f'{symbol} hissesi piyasa performansı',
            f'{symbol} için uzman görüşleri'
        ])
        
        # Tarih ve sentiment ekle
        news_list = []
        for i, headline in enumerate(headlines[:limit]):
            news_list.append({
                'headline': headline,
                'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                'sentiment': round(self._calculate_mock_sentiment(symbol) + (i * 0.05), 2),
                'source': 'Demo News'
            })
        
        return news_list
    
    def analyze_social_media_sentiment(self, symbol: str) -> Dict:
        """Sosyal medya sentiment analizi (mock)"""
        
        base_sentiment = self._calculate_mock_sentiment(symbol)
        
        # Sosyal medya genelde daha volatil
        import random
        random.seed(int(time.time()) // 1800)  # 30 dakikalık değişim
        social_multiplier = random.uniform(0.8, 1.5)
        
        social_sentiment = base_sentiment * social_multiplier
        social_sentiment = max(-1.0, min(1.0, social_sentiment))
        
        return {
            'symbol': symbol,
            'social_sentiment': round(social_sentiment, 2),
            'social_label': self._get_sentiment_label(social_sentiment),
            'mention_count': random.randint(50, 500),
            'engagement_rate': round(random.uniform(0.02, 0.08), 3),
            'trending_status': 'Trending' if abs(social_sentiment) > 0.3 else 'Normal'
        }
    
    def get_sentiment_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Sentiment geçmişi (mock veri)"""
        
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        sentiments = []
        
        base_sentiment = self._calculate_mock_sentiment(symbol)
        
        for i, date in enumerate(dates):
            # Günlük sentiment değişimi
            import random
            random.seed(int(date.timestamp()) // 86400)
            daily_sentiment = base_sentiment + random.uniform(-0.2, 0.2)
            daily_sentiment = max(-1.0, min(1.0, daily_sentiment))
            
            sentiments.append({
                'date': date.strftime('%Y-%m-%d'),
                'sentiment': round(daily_sentiment, 2),
                'news_count': random.randint(1, 10),
                'social_mentions': random.randint(20, 200)
            })
        
        return pd.DataFrame(sentiments).sort_values('date')
    
    def _fetch_rss(self, source_name: str, rss_url: str):
        """Fetch bytes with the accepted request budget, then parse once."""
        started = time.monotonic()
        diagnostic = {
            'source': source_name, 'status': 'failed', 'attempts': 0,
            'elapsed_seconds': 0.0, 'error_code': None,
        }
        feed = None
        for attempt in range(1, 4):
            diagnostic['attempts'] = attempt
            response = None
            retryable = False
            body = None
            try:
                response = requests.get(rss_url, timeout=(8, 8))
                status = response.status_code
                if 200 <= status < 300:
                    body = response.content
                elif status in (429, 503) and 'Retry-After' in response.headers:
                    diagnostic['error_code'] = 'retry_after_deferred'
                else:
                    diagnostic['error_code'] = 'http_error'
                    retryable = status in (408, 429, 500, 502, 503, 504)
            except requests.Timeout:
                diagnostic['error_code'] = 'timeout'
                retryable = True
            except requests.ConnectionError:
                diagnostic['error_code'] = 'connection_error'
                retryable = True
            except Exception:
                diagnostic['error_code'] = 'fetch_failed'
            finally:
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        diagnostic['error_code'] = 'fetch_failed'
                        retryable = False
                        body = None

            if body is not None:
                # Parsing is outside the transport retry boundary.
                try:
                    feed = feedparser.parse(body)
                    has_entries = bool(feed.entries)
                    if getattr(feed, 'bozo', False):
                        diagnostic['status'] = 'partial' if has_entries else 'failed'
                        diagnostic['error_code'] = 'parse_partial' if has_entries else 'parse_failed'
                    else:
                        diagnostic['status'] = 'success' if has_entries else 'empty'
                        diagnostic['error_code'] = None
                except Exception:
                    diagnostic['error_code'] = 'parse_failed'
                    feed = None
                break
            if not retryable or attempt == 3:
                break
            time.sleep(1.5 * attempt)

        diagnostic['elapsed_seconds'] = max(0.0, time.monotonic() - started)
        return feed, diagnostic

    def fetch_real_news(self, limit: int = 20) -> List[Dict]:
        """Gerçek RSS feed'lerden haber çeker"""
        all_news = []
        self.last_fetch_diagnostics = []
        
        for source_name, rss_url in self.news_sources.items():
            try:
                feed, diagnostic = self._fetch_rss(source_name, rss_url)
                self.last_fetch_diagnostics.append(diagnostic)
                if feed is None:
                    continue
                fetched_at = datetime.now(timezone.utc).isoformat()
                
                for entry in feed.entries[:limit//len(self.news_sources)]:
                    # Haber detaylarını al
                    title = entry.get('title', '')
                    description = entry.get('description', '')
                    link = entry.get('link', '')
                    published = entry.get('published', '')
                    
                    # Kaynak zamanı bilinmiyorsa çekilme zamanı ile doldurma.
                    published_at = None
                    formatted_date = ''
                    published_time_status = 'missing' if published in (None, '') else 'invalid'
                    if isinstance(published, str) and published:
                        try:
                            # Önceki parser'ın kabul ettiği +03:00 biçimini de koru.
                            rfc_date = re.sub(r'([+-]\d{2}):(\d{2})$', r'\1\2', published)
                            pub_date = parsedate_to_datetime(rfc_date)
                            if pub_date.tzinfo is not None and pub_date.utcoffset() is not None:
                                published_at = pub_date.isoformat()
                                formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')
                                published_time_status = 'valid'
                        except (ValueError, TypeError, IndexError, OverflowError):
                            # Bozuk kaynak tarihi haberin kaybolmasına neden olmaz.
                            pass
                    
                    # Sentiment analizi yap
                    text_for_analysis = f"{title} {description}"
                    sentiment_score = self._analyze_text_sentiment(text_for_analysis)
                    
                    # Hisse sembolü tespit et
                    detected_symbols = self._detect_symbols_in_text(text_for_analysis)
                    
                    news_item = {
                        'title': title,
                        'description': description,
                        'link': link,
                        'date': formatted_date,
                        'published_at': published_at,
                        'published_time_status': published_time_status,
                        'fetched_at': fetched_at,
                        'source': source_name.replace('_', ' ').title(),
                        'sentiment_status': 'computed' if sentiment_score is not None else 'failed',
                        'sentiment_score': sentiment_score,
                        'sentiment_label': self._get_sentiment_label(sentiment_score) if sentiment_score is not None else None,
                        'detected_symbols': detected_symbols
                    }
                    
                    all_news.append(news_item)
                    
            except Exception as e:
                print(f"RSS feed hatası ({source_name}): {str(e)}")
                continue
        
        # Farklı offset'leri aynı zaman ekseninde karşılaştır; bilinmeyenler sonda.
        all_news.sort(
            key=lambda item: datetime.fromisoformat(item['published_at']).astimezone(timezone.utc)
            if item['published_at'] is not None else datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return all_news[:limit]
    
    def _analyze_text_sentiment(self, text: str) -> Optional[float]:
        """Metin sentiment analizi yapar; hesaplama hatasında skor üretmez."""
        try:
            # VADER sentiment analizi (İngilizce metinler için)
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # TextBlob sentiment analizi
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # İki yöntemin ortalamasını al
            final_sentiment = (vader_compound + textblob_polarity) / 2
            if not isfinite(final_sentiment):
                return None
            
            return round(final_sentiment, 3)
            
        except Exception as e:
            print(f"Sentiment analizi hatası: {str(e)}")
            return None
    
    def _detect_symbols_in_text(self, text: str) -> List[str]:
        """Metinde hisse sembollerini tespit eder"""
        detected = []
        text_upper = text.upper()
        
        for symbol in self.symbol_sentiments.keys():
            # Sembol adını ve şirket adını ara
            if symbol in text_upper:
                detected.append(symbol)
        
        return detected
    
    def get_symbol_specific_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Belirli bir hisse için haberleri filtreler"""
        all_news = self.fetch_real_news(limit=50)
        symbol_news = []
        
        for news in all_news:
            # Haber başlığında veya açıklamasında sembol geçiyor mu?
            text_to_search = f"{news['title']} {news['description']}".upper()
            
            if symbol.upper() in text_to_search or symbol in news['detected_symbols']:
                symbol_news.append(news)
                
            if len(symbol_news) >= limit:
                break
        
        return symbol_news
    
    def get_market_news_summary(self) -> Dict:
        """Piyasa geneli haber özeti"""
        try:
            news = self.fetch_real_news(limit=30)
            source_diagnostics = [dict(item) for item in self.last_fetch_diagnostics]
            
            # Sentiment istatistikleri
            sentiments = [n['sentiment_score'] for n in news
                          if n['sentiment_status'] == 'computed' and n['sentiment_score'] is not None]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
            
            positive_count = len([s for s in sentiments if s > 0.1])
            negative_count = len([s for s in sentiments if s < -0.1])
            neutral_count = len(sentiments) - positive_count - negative_count
            
            # En çok geçen semboller
            symbol_mentions = {}
            for news_item in news:
                for symbol in news_item['detected_symbols']:
                    symbol_mentions[symbol] = symbol_mentions.get(symbol, 0) + 1
            
            top_symbols = sorted(symbol_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_news': len(news),
                'scored_news': len(sentiments),
                'failed_sentiment_count': sum(n['sentiment_status'] == 'failed' for n in news),
                'avg_sentiment': round(avg_sentiment, 3) if avg_sentiment is not None else None,
                'sentiment_label': self._get_sentiment_label(avg_sentiment) if avg_sentiment is not None else None,
                'positive_news': positive_count,
                'negative_news': negative_count,
                'neutral_news': neutral_count,
                'top_symbols': [{'symbol': s[0], 'mentions': s[1]} for s in top_symbols],
                'recent_news': news[:5],
                'source_diagnostics': source_diagnostics,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'total_news': 0,
                'scored_news': 0,
                'failed_sentiment_count': 0,
                'avg_sentiment': None,
                'sentiment_label': None,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'top_symbols': [],
                'recent_news': [],
                'source_diagnostics': [dict(item) for item in self.last_fetch_diagnostics],
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
