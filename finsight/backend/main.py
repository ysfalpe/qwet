from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import os
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import time
import logging
from pydantic import BaseModel, Field

# .env dosyasından çevresel değişkenleri yükle
load_dotenv()

# Loglamayı yapılandır
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Finsight Backend API", 
              description="Finnhub API ve OpenRouter API için önbellekleme ve proxy servisi",
              version="1.0.0")

# CORS ayarları (Flutter web uygulaması için gerekli)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin ver (geliştirme için)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API anahtarları
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Yüklenen anahtarları (maskelenmiş olarak) logla
logger.info(f"FINNHUB_API_KEY loaded: {FINNHUB_API_KEY[:5]}..." if FINNHUB_API_KEY else "FINNHUB_API_KEY not found!")
logger.info(f"OPENROUTER_API_KEY loaded: {OPENROUTER_API_KEY[:5]}..." if OPENROUTER_API_KEY else "OPENROUTER_API_KEY not found!")

# Anahtarların varlığını kontrol et
if not FINNHUB_API_KEY:
    logger.error("FINNHUB_API_KEY ortam değişkeni bulunamadı!")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY ortam değişkeni bulunamadı!")

# API URL'leri
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Önbellek (gerçek uygulamada Redis veya veritabanı kullanılabilir)
cache = {
    "stock_quotes": {},       # Hisse senedi fiyatları
    "company_profiles": {},   # Şirket profilleri
    "company_news": {},       # Şirket haberleri
    "market_news": [],        # Piyasa haberleri
    "search_results": {},     # Arama sonuçları
    "ai_analysis": {},        # AI analizleri
    "last_updated": {}        # Son güncelleme zamanları
}

# Popüler hisse senetleri
POPULAR_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']

# Finnhub API'den hisse fiyatlarını çekme
def fetch_stock_quotes():
    logger.info(f"Hisse fiyatları güncelleniyor: {datetime.now()}")
    if not FINNHUB_API_KEY:
        logger.error("Finnhub API anahtarı eksik, hisse fiyatları çekilemiyor.")
        return

    current_quotes = {}
    for symbol in POPULAR_STOCKS:
        try:
            response = requests.get(
                f"{FINNHUB_BASE_URL}/quote?symbol={symbol}&token={FINNHUB_API_KEY}",
                timeout=10
            )
            response.raise_for_status()
            current_quotes[symbol] = response.json()
            logger.info(f"{symbol} fiyatı güncellendi")
        except requests.exceptions.RequestException as e:
            logger.error(f"{symbol} fiyatı güncellenemedi (Request Hatası): {e}")
        except Exception as e:
            logger.error(f"{symbol} fiyatı için genel hata: {e}")

    # Önbelleğe yazmadan hemen önce log ekle
    logger.info(f"fetch_stock_quotes - Önbelleğe yazılacak {len(current_quotes)} adet hisse verisi hazır.")
    
    cache["popular_stocks"] = current_quotes
    
    # Önbelleğe yazdıktan hemen sonra log ekle
    logger.info(f"fetch_stock_quotes - Popüler hisse verileri cache['popular_stocks']'a yazıldı.")

    # last_updated zaman damgasını DOĞRU anahtar için güncelle
    if "last_updated" not in cache: # Bu kontrol zaten yukarıda vardı ama tekrar ekleyelim
        cache["last_updated"] = {}
    cache["last_updated"]["popular_stocks"] = datetime.now() # 'stock_quotes' yerine 'popular_stocks'

# Şirket profillerini çekme
def fetch_company_profiles():
    logger.info(f"Şirket profilleri güncelleniyor: {datetime.now()}")
    if not FINNHUB_API_KEY:
        logger.error("Finnhub API anahtarı eksik, şirket profilleri çekilemiyor.")
        return

    for symbol in POPULAR_STOCKS:
        try:
            response = requests.get(
                f"{FINNHUB_BASE_URL}/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}",
                timeout=10
            )
            response.raise_for_status()
            cache["company_profiles"][symbol] = response.json()
            logger.info(f"{symbol} profili güncellendi")
        except requests.exceptions.RequestException as e:
            logger.error(f"{symbol} profili güncellenemedi (Request Hatası): {e}")
        except Exception as e:
            logger.error(f"{symbol} profili için genel hata: {e}")

    cache["last_updated"]["company_profiles"] = datetime.now()

# Piyasa haberlerini çekme
def fetch_market_news():
    logger.info(f"Piyasa haberleri güncelleniyor: {datetime.now()}")
    if not FINNHUB_API_KEY:
        logger.error("Finnhub API anahtarı eksik, piyasa haberleri çekilemiyor.")
        return

    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/news?category=general&token={FINNHUB_API_KEY}",
            timeout=10
        )
        response.raise_for_status()
        cache["market_news"] = response.json()
        logger.info("Piyasa haberleri güncellendi")
    except requests.exceptions.RequestException as e:
        logger.error(f"Piyasa haberleri güncellenemedi (Request Hatası): {e}")
    except Exception as e:
        logger.error(f"Piyasa haberleri için genel hata: {e}")

    cache["last_updated"]["market_news"] = datetime.now()

# Zamanlanmış görevleri başlatma
@app.on_event("startup")
def start_scheduler():
    logger.info("Zamanlayıcı başlatılıyor...")
    scheduler = BackgroundScheduler(timezone="UTC") # Zaman dilimi belirtmek iyi olabilir
    # Her 1 dakikada bir hisse fiyatlarını güncelle
    scheduler.add_job(fetch_stock_quotes, 'interval', minutes=1, id="fetch_quotes_job")
    # Her saat başı şirket profillerini güncelle (daha sık kontrol için)
    scheduler.add_job(fetch_company_profiles, 'interval', hours=1, id="fetch_profiles_job")
    # Her 15 dakikada bir haberleri güncelle (daha sık kontrol için)
    scheduler.add_job(fetch_market_news, 'interval', minutes=15, id="fetch_news_job")
    
    try:
        scheduler.start()
        logger.info("Zamanlayıcı başarıyla başlatıldı.")
    except Exception as e:
        logger.critical(f"Zamanlayıcı başlatılamadı: {e}", exc_info=True)

# API endpoint'leri

@app.get("/")
async def root():
    return {"message": "Finsight Backend API çalışıyor"}

@app.get("/api/stock/quote")
async def get_stock_quote(symbol: str):
    """
    Hisse senedi fiyat bilgilerini döndürür.
    Önbellekte varsa ve son 5 dakika içinde güncellendiyse, önbellekten döndürür.
    """
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="Finnhub API anahtarı yapılandırılmamış.")

    # Önbellekte varsa ve son 5 dakika içinde güncellendiyse, önbellekten döndür
    if (symbol in cache["stock_quotes"] and 
        "stock_quotes" in cache["last_updated"] and
        datetime.now() - cache["last_updated"]["stock_quotes"] < timedelta(minutes=5)):
        return cache["stock_quotes"][symbol]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/quote?symbol={symbol}&token={FINNHUB_API_KEY}",
            timeout=10 # Timeout zaten eklenmişti, kontrol
        )
        response.raise_for_status() # HTTP hatası varsa exception fırlat
        data = response.json()
        cache["stock_quotes"][symbol] = data
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"/api/stock/quote - {symbol} için Finnhub API hatası: {e}") # Endpoint loglaması
        raise HTTPException(status_code=e.response.status_code if e.response else 502, detail=f"Finnhub API'den veri alınamadı: {e}") # 502 Bad Gateway
    except Exception as e:
        logger.error(f"/api/stock/quote - {symbol} için beklenmedik hata: {e}", exc_info=True) # Traceback logla
        raise HTTPException(status_code=500, detail=f"İç sunucu hatası.") # Detayı gizle

@app.get("/api/stock/profile")
async def get_company_profile(symbol: str):
    """
    Şirket profil bilgilerini döndürür.
    Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndürür.
    """
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="Finnhub API anahtarı yapılandırılmamış.")

    # Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndür
    if (symbol in cache["company_profiles"] and 
        "company_profiles" in cache["last_updated"] and
        datetime.now() - cache["last_updated"]["company_profiles"] < timedelta(hours=24)):
        return cache["company_profiles"][symbol]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        cache["company_profiles"][symbol] = data
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"/api/stock/profile - {symbol} için Finnhub API hatası: {e}")
        raise HTTPException(status_code=e.response.status_code if e.response else 502, detail=f"Finnhub API'den veri alınamadı: {e}")
    except Exception as e:
        logger.error(f"/api/stock/profile - {symbol} için beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="İç sunucu hatası.")

@app.get("/api/news/company")
async def get_company_news(symbol: str, count: int = 10):
    """
    Şirket haberlerini döndürür.
    Önbellekte varsa ve son 30 dakika içinde güncellendiyse, önbellekten döndürür.
    """
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="Finnhub API anahtarı yapılandırılmamış.")

    # Önbellekte varsa ve son 30 dakika içinde güncellendiyse, önbellekten döndür
    if (symbol in cache["company_news"] and 
        f"company_news_{symbol}" in cache["last_updated"] and
        datetime.now() - cache["last_updated"][f"company_news_{symbol}"] < timedelta(minutes=30)):
        return cache["company_news"][symbol][:count]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        # Son 30 günün haberlerini al
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        from_date = thirty_days_ago.strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")
        
        response = requests.get(
            f"{FINNHUB_BASE_URL}/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}",
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        cache["company_news"][symbol] = data
        cache["last_updated"][f"company_news_{symbol}"] = datetime.now()
        return data[:count]
    except requests.exceptions.RequestException as e:
        logger.error(f"/api/news/company - {symbol} için Finnhub API hatası: {e}")
        raise HTTPException(status_code=e.response.status_code if e.response else 502, detail=f"Finnhub API'den veri alınamadı: {e}")
    except Exception as e:
        logger.error(f"/api/news/company - {symbol} için beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="İç sunucu hatası.")

@app.get("/api/news/market")
async def get_market_news(count: int = 10):
    """
    Piyasa haberlerini döndürür.
    Önbellekte varsa ve son 30 dakika içinde güncellendiyse, önbellekten döndürür.
    """
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="Finnhub API anahtarı yapılandırılmamış.")

    # Önbellekte varsa ve son 30 dakika içinde güncellendiyse, önbellekten döndür
    if (cache["market_news"] and 
        "market_news" in cache["last_updated"] and
        datetime.now() - cache["last_updated"]["market_news"] < timedelta(minutes=30)):
        return cache["market_news"][:count]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/news?category=general&token={FINNHUB_API_KEY}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        cache["market_news"] = data
        cache["last_updated"]["market_news"] = datetime.now()
        return data[:count]
    except requests.exceptions.RequestException as e:
        logger.error(f"/api/news/market için Finnhub API hatası: {e}")
        raise HTTPException(status_code=e.response.status_code if e.response else 502, detail=f"Finnhub API'den veri alınamadı: {e}")
    except Exception as e:
        logger.error(f"/api/news/market için beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="İç sunucu hatası.")

@app.get("/api/search")
async def search_stocks(query: str):
    """
    Hisse senedi arama sonuçlarını döndürür.
    Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndürür.
    """
    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="Finnhub API anahtarı yapılandırılmamış.")

    # Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndür
    if (query in cache["search_results"] and 
        f"search_{query}" in cache["last_updated"] and
        datetime.now() - cache["last_updated"][f"search_{query}"] < timedelta(hours=24)):
        return cache["search_results"][query]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/search?q={query}&token={FINNHUB_API_KEY}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Popüler ABD hisseleri için özel durum
        results = data.get('result', [])
        
        # Popüler hisseler sözlüğü
        popular_stocks = {
            'apple': {'symbol': 'AAPL', 'description': 'Apple Inc.'},
            'microsoft': {'symbol': 'MSFT', 'description': 'Microsoft Corporation'},
            'google': {'symbol': 'GOOGL', 'description': 'Alphabet Inc.'},
            'amazon': {'symbol': 'AMZN', 'description': 'Amazon.com Inc.'},
            'tesla': {'symbol': 'TSLA', 'description': 'Tesla, Inc.'},
            'meta': {'symbol': 'META', 'description': 'Meta Platforms, Inc.'},
            'netflix': {'symbol': 'NFLX', 'description': 'Netflix, Inc.'},
            'nvidia': {'symbol': 'NVDA', 'description': 'NVIDIA Corporation'},
            'disney': {'symbol': 'DIS', 'description': 'The Walt Disney Company'},
            'coca': {'symbol': 'KO', 'description': 'The Coca-Cola Company'},
            'pepsi': {'symbol': 'PEP', 'description': 'PepsiCo, Inc.'},
            'walmart': {'symbol': 'WMT', 'description': 'Walmart Inc.'},
            'mcdonalds': {'symbol': 'MCD', 'description': 'McDonald\'s Corporation'},
            'nike': {'symbol': 'NKE', 'description': 'NIKE, Inc.'},
            'intel': {'symbol': 'INTC', 'description': 'Intel Corporation'},
            'amd': {'symbol': 'AMD', 'description': 'Advanced Micro Devices, Inc.'},
            'ford': {'symbol': 'F', 'description': 'Ford Motor Company'},
            'gm': {'symbol': 'GM', 'description': 'General Motors Company'},
            'boeing': {'symbol': 'BA', 'description': 'The Boeing Company'},
            'visa': {'symbol': 'V', 'description': 'Visa Inc.'},
            'mastercard': {'symbol': 'MA', 'description': 'Mastercard Incorporated'},
            'paypal': {'symbol': 'PYPL', 'description': 'PayPal Holdings, Inc.'},
            'bank of america': {'symbol': 'BAC', 'description': 'Bank of America Corporation'},
            'jpmorgan': {'symbol': 'JPM', 'description': 'JPMorgan Chase & Co.'},
            'goldman': {'symbol': 'GS', 'description': 'The Goldman Sachs Group, Inc.'},
        }
        
        # Eğer arama sorgusu popüler bir hisse ile eşleşiyorsa, onu listenin başına ekle
        for entry_key, entry_value in popular_stocks.items():
            if (query.lower() in entry_key or 
                entry_value['description'].lower().find(query.lower()) != -1 or
                entry_value['symbol'].lower() == query.lower()):
                
                # Eğer bu hisse zaten sonuçlarda yoksa ekle
                if not any(result.get('symbol') == entry_value['symbol'] for result in results):
                    results.insert(0, {
                        'symbol': entry_value['symbol'],
                        'description': entry_value['description'],
                        'type': 'Common Stock',
                        'displaySymbol': entry_value['symbol']
                    })
        
        # ABD borsalarındaki hisseleri önceliklendirme
        filtered_results = [
            result for result in results
            if not result.get('symbol', '').find('.') != -1 or 
               result.get('symbol', '').endswith('.US') or 
               any(exchange in result.get('description', '').upper() 
                   for exchange in ['NYSE', 'NASDAQ', 'AMEX'])
        ]
        
        cache["search_results"][query] = filtered_results
        cache["last_updated"][f"search_{query}"] = datetime.now()
        return filtered_results
    except requests.exceptions.RequestException as e:
        logger.error(f"/api/search - '{query}' için Finnhub API hatası: {e}")
        raise HTTPException(status_code=e.response.status_code if e.response else 502, detail=f"Finnhub API'den veri alınamadı: {e}")
    except Exception as e:
        logger.error(f"/api/search - '{query}' için beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="İç sunucu hatası.")

@app.get("/api/popular-stocks")
async def get_popular_stocks():
    """Popüler hisse senetlerinin güncel fiyatlarını döndürür. Eksik veya eski veri varsa bile mevcut olanı döndürmeye çalışır."""
    global cache
    required_keys = {'c', 'd', 'dp', 'h', 'l', 'o', 'pc', 't'} # Finnhub /quote'dan beklenen temel anahtarlar
    result_stocks = []
    data_is_potentially_stale = False

    logger.info(f"/api/popular-stocks - Çağrı başlangıcı. Önbellek durumu: popular_stocks var mı? {'popular_stocks' in cache}, last_updated var mı? {'last_updated' in cache and 'popular_stocks' in cache.get('last_updated', {})}")
    if 'popular_stocks' in cache:
         logger.debug(f"/api/popular-stocks - Önbellekteki popüler hisse anahtarları: {list(cache['popular_stocks'].keys())}") # Sadece anahtarları logla

    # Genel veri yaşı kontrolü (Sadece uyarı için, 10 dk'dan eski ise)
    if not ("last_updated" in cache and "popular_stocks" in cache["last_updated"] and
            datetime.now() - cache["last_updated"]["popular_stocks"] < timedelta(minutes=10)):
        logger.warning("/api/popular-stocks - Genel popüler hisse senedi verisi 10 dakikadan eski veya zaman damgası eksik.")
        data_is_potentially_stale = True # Bu aslında bir hata değil, sadece bilgi

    if "popular_stocks" in cache:
        for symbol in POPULAR_STOCKS:
            if symbol in cache["popular_stocks"]:
                stock_cache_entry = cache["popular_stocks"][symbol]
                if isinstance(stock_cache_entry, dict) and required_keys.issubset(stock_cache_entry.keys()):
                    # Sembol için veri varsa ve gerekli anahtarları içeriyorsa ekle
                    stock_data = stock_cache_entry.copy() # Kopyasını alarak çalışmak daha güvenli
                    stock_data['symbol'] = symbol # Sembolü veriye ekleyelim frontend için kolaylık
                    result_stocks.append(stock_data)
                    # logger.debug(f"/api/popular-stocks - {symbol} listeye eklendi.") # Çok fazla log olmaması için yorumda
                else:
                    # Sembol için veri var ama eksik veya yanlış formatta
                    actual_keys = stock_cache_entry.keys() if isinstance(stock_cache_entry, dict) else "Veri dict değil"
                    logger.warning(f"/api/popular-stocks - {symbol} için önbellekte veri var ama gerekli anahtarlar eksik/yanlış formatta. Beklenen: {required_keys}, Bulunan: {actual_keys}")
            else:
                # Sembol için veri yoksa logla (ama hata verme)
                logger.warning(f"/api/popular-stocks - {symbol} için önbellekte veri yok.")

    if not result_stocks:
        # Eğer *hiç* veri bulunamazsa (örn. uygulama yeni başladıysa veya tümü eksikse), o zaman hata verelim.
        logger.error("/api/popular-stocks - Önbellekte gösterilecek hiç popüler hisse senedi verisi bulunamadı (result_stocks boş). Cache popular_stocks içeriği (ilk 5): {str(cache.get('popular_stocks', {}))[:500]}")
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Popüler hisse senedi verileri şu anda alınamıyor veya eksik. Lütfen birkaç dakika sonra tekrar deneyin."
        )

    logger.info(f"/api/popular-stocks - {len(result_stocks)} adet popüler hisse senedi döndürülüyor.")
    # İsteğe bağlı: Yanıta 'is_stale': data_is_potentially_stale eklenebilir.
    return {"stocks": result_stocks}

# Pydantic model for AI Analysis request
class AnalysisRequest(BaseModel):
    symbol: str
    company_name: str
    price: float
    change: float

@app.post("/api/analysis")
async def get_ai_analysis(request_data: AnalysisRequest):
    """
    Hisse senedi için AI analizi döndürür.
    Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndürür.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OpenRouter API anahtarı yapılandırılmamış.")

    symbol = request_data.symbol
    company_name = request_data.company_name
    price = request_data.price
    change = request_data.change

    cache_key = f"{symbol}_{price}_{change}"
    
    # Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndür
    if (cache_key in cache["ai_analysis"] and 
        f"ai_analysis_{cache_key}" in cache["last_updated"] and
        datetime.now() - cache["last_updated"][f"ai_analysis_{cache_key}"] < timedelta(hours=24)):
        return {"analysis": cache["ai_analysis"][cache_key]}
    
    # Yoksa veya güncel değilse, OpenRouter'dan çek
    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'HTTP-Referer': 'https://finsight.app',
            },
            json={
                'model': 'deepseek/deepseek-chat-v3-0324',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a financial analyst AI assistant. Provide a brief, professional analysis of the stock based on the information provided. Focus on recent performance, potential factors affecting the stock, and a very brief outlook. Keep your response under 200 words and focus on facts rather than speculation.'
                    },
                    {
                        'role': 'user',
                        'content': f'Provide a brief analysis for {company_name} ({symbol}). Current price: ${price}, Change: {"+" + str(change) if change > 0 else str(change)}'
                    }
                ],
                'max_tokens': 300,
            }
        )

        logger.info(f"/api/analysis - OpenRouter API yanıt alındı (Sembol: {symbol}): {response.text[:200]}...") # Yanıt çok uzunsa kısaltarak logla

        if response.status_code == 200:
            result = response.json()
            analysis = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            cache["ai_analysis"][cache_key] = analysis
            cache["last_updated"][f"ai_analysis_{cache_key}"] = datetime.now()
            logger.info(f"/api/analysis - Başarılı yanıt gönderiliyor (Sembol: {symbol})")
            return {"analysis": analysis}
        else:
            logger.error(f"/api/analysis - OpenRouter API hatası: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API hatası: {response.text}")
    except Exception as e:
        logger.error(f"/api/analysis - Beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI analizi alınırken hata oluştu.")

# Uygulama durumu endpoint'i
@app.get("/api/status")
async def get_status():
    """
    Backend uygulamasının durumunu ve önbellek bilgilerini döndürür.
    """
    status = {
        "status": "running",
        "uptime": time.time(),
        "cache_info": {
            "stock_quotes_count": len(cache["stock_quotes"]),
            "company_profiles_count": len(cache["company_profiles"]),
            "company_news_count": len(cache["company_news"]),
            "market_news_count": len(cache["market_news"]),
            "search_results_count": len(cache["search_results"]),
            "ai_analysis_count": len(cache["ai_analysis"]),
        },
        "last_updated": {k: v.isoformat() if isinstance(v, datetime) else str(v) 
                         for k, v in cache["last_updated"].items()}
    }
    return status

# Uygulamayı çalıştırma (doğrudan bu dosya çalıştırıldığında)
if __name__ == "__main__":
    import uvicorn
    # import os # Zaten yukarıda import edildi
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Uvicorn sunucusu {port} portunda başlatılıyor...")
    # Gunicorn yerine Uvicorn ile çalıştırırken reload flag'ı geliştirme içindir
    # Cloud Run'da gunicorn kullanıldığı için bu kısım sadece lokal test içindir
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False) # Cloud Run için reload=False olmalı
