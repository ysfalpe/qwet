from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import os
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import time

# .env dosyasından çevresel değişkenleri yükle
load_dotenv()

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

# API URL'leri
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Önbellek (gerçek uygulamada Redis veya veritabanı kullanılabilir)
cache = {
    "stock_quotes": {},       # Hisse senedi fiyatları
    "company_profiles": {},   # Şirket profilleri
    "stock_candles": {},      # Grafik verileri
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
    print(f"Hisse fiyatları güncelleniyor: {datetime.now()}")
    for symbol in POPULAR_STOCKS:
        try:
            response = requests.get(
                f"{FINNHUB_BASE_URL}/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
            )
            if response.status_code == 200:
                cache["stock_quotes"][symbol] = response.json()
                print(f"{symbol} fiyatı güncellendi")
            else:
                print(f"{symbol} fiyatı güncellenemedi: {response.status_code}")
                
        except Exception as e:
            print(f"{symbol} için hata: {e}")
    
    cache["last_updated"]["stock_quotes"] = datetime.now()

# Şirket profillerini çekme
def fetch_company_profiles():
    print(f"Şirket profilleri güncelleniyor: {datetime.now()}")
    for symbol in POPULAR_STOCKS:
        try:
            response = requests.get(
                f"{FINNHUB_BASE_URL}/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}"
            )
            if response.status_code == 200:
                cache["company_profiles"][symbol] = response.json()
                print(f"{symbol} profili güncellendi")
            else:
                print(f"{symbol} profili güncellenemedi: {response.status_code}")
                
        except Exception as e:
            print(f"{symbol} profili için hata: {e}")
    
    cache["last_updated"]["company_profiles"] = datetime.now()

# Piyasa haberlerini çekme
def fetch_market_news():
    print(f"Piyasa haberleri güncelleniyor: {datetime.now()}")
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/news?category=general&token={FINNHUB_API_KEY}"
        )
        if response.status_code == 200:
            cache["market_news"] = response.json()
            print("Piyasa haberleri güncellendi")
        else:
            print(f"Piyasa haberleri güncellenemedi: {response.status_code}")
            
    except Exception as e:
        print(f"Piyasa haberleri için hata: {e}")
    
    cache["last_updated"]["market_news"] = datetime.now()

# Zamanlanmış görevleri başlatma
@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    # Her 5 dakikada bir hisse fiyatlarını güncelle
    scheduler.add_job(fetch_stock_quotes, 'interval', minutes=5)
    # Her gün şirket profillerini güncelle
    scheduler.add_job(fetch_company_profiles, 'interval', hours=24)
    # Her 30 dakikada bir haberleri güncelle
    scheduler.add_job(fetch_market_news, 'interval', minutes=30)
    scheduler.start()
    
    # İlk verileri hemen çek
    fetch_stock_quotes()
    fetch_company_profiles()
    fetch_market_news()

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
    # Önbellekte varsa ve son 5 dakika içinde güncellendiyse, önbellekten döndür
    if (symbol in cache["stock_quotes"] and 
        "stock_quotes" in cache["last_updated"] and
        datetime.now() - cache["last_updated"]["stock_quotes"] < timedelta(minutes=5)):
        return cache["stock_quotes"][symbol]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
        )
        if response.status_code == 200:
            data = response.json()
            cache["stock_quotes"][symbol] = data
            return data
        else:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to fetch quote: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/profile")
async def get_company_profile(symbol: str):
    """
    Şirket profil bilgilerini döndürür.
    Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndürür.
    """
    # Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndür
    if (symbol in cache["company_profiles"] and 
        "company_profiles" in cache["last_updated"] and
        datetime.now() - cache["last_updated"]["company_profiles"] < timedelta(hours=24)):
        return cache["company_profiles"][symbol]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}"
        )
        if response.status_code == 200:
            data = response.json()
            cache["company_profiles"][symbol] = data
            return data
        else:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to fetch company profile: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/candles")
async def get_stock_candles(symbol: str, resolution: str, from_time: int, to_time: int):
    """
    Hisse senedi grafik verilerini döndürür.
    Önbellekte varsa ve sorgu parametreleri aynıysa, önbellekten döndürür.
    """
    cache_key = f"{symbol}_{resolution}_{from_time}_{to_time}"
    
    # Önbellekte varsa ve aynı sorgu parametreleriyle, önbellekten döndür
    if cache_key in cache["stock_candles"]:
        return cache["stock_candles"][cache_key]
    
    # Yoksa, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/stock/candle?symbol={symbol}&resolution={resolution}&from={from_time}&to={to_time}&token={FINNHUB_API_KEY}"
        )
        if response.status_code == 200:
            data = response.json()
            cache["stock_candles"][cache_key] = data
            return data
        else:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to fetch stock candles: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news/company")
async def get_company_news(symbol: str, count: int = 10):
    """
    Şirket haberlerini döndürür.
    Önbellekte varsa ve son 30 dakika içinde güncellendiyse, önbellekten döndürür.
    """
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
            f"{FINNHUB_BASE_URL}/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={FINNHUB_API_KEY}"
        )
        if response.status_code == 200:
            data = response.json()
            cache["company_news"][symbol] = data
            cache["last_updated"][f"company_news_{symbol}"] = datetime.now()
            return data[:count]
        else:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to fetch company news: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news/market")
async def get_market_news(count: int = 10):
    """
    Piyasa haberlerini döndürür.
    Önbellekte varsa ve son 30 dakika içinde güncellendiyse, önbellekten döndürür.
    """
    # Önbellekte varsa ve son 30 dakika içinde güncellendiyse, önbellekten döndür
    if (cache["market_news"] and 
        "market_news" in cache["last_updated"] and
        datetime.now() - cache["last_updated"]["market_news"] < timedelta(minutes=30)):
        return cache["market_news"][:count]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/news?category=general&token={FINNHUB_API_KEY}"
        )
        if response.status_code == 200:
            data = response.json()
            cache["market_news"] = data
            cache["last_updated"]["market_news"] = datetime.now()
            return data[:count]
        else:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to fetch market news: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
async def search_stocks(query: str):
    """
    Hisse senedi arama sonuçlarını döndürür.
    Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndürür.
    """
    # Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndür
    if (query in cache["search_results"] and 
        f"search_{query}" in cache["last_updated"] and
        datetime.now() - cache["last_updated"][f"search_{query}"] < timedelta(hours=24)):
        return cache["search_results"][query]
    
    # Yoksa veya güncel değilse, Finnhub'dan çek
    try:
        response = requests.get(
            f"{FINNHUB_BASE_URL}/search?q={query}&token={FINNHUB_API_KEY}"
        )
        if response.status_code == 200:
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
        else:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to search stocks: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/popular-stocks")
async def get_popular_stocks():
    """
    Popüler hisse senetlerinin fiyat ve profil bilgilerini döndürür.
    Önbellekte varsa ve son 5 dakika içinde güncellendiyse, önbellekten döndürür.
    """
    # Önbellekte varsa ve son 5 dakika içinde güncellendiyse, önbellekten döndür
    if ("stock_quotes" in cache["last_updated"] and
        datetime.now() - cache["last_updated"]["stock_quotes"] < timedelta(minutes=5)):
        
        result = []
        for symbol in POPULAR_STOCKS:
            if symbol in cache["stock_quotes"] and symbol in cache["company_profiles"]:
                quote = cache["stock_quotes"][symbol]
                profile = cache["company_profiles"][symbol]
                
                result.append({
                    "symbol": symbol,
                    "name": profile.get("name", ""),
                    "price": quote.get("c", 0),
                    "change": quote.get("d", 0),
                    "changePercent": quote.get("dp", 0),
                    "logo": profile.get("logo", "")
                })
        
        return result
    
    # Yoksa veya güncel değilse, verileri güncelle ve döndür
    fetch_stock_quotes()
    fetch_company_profiles()
    
    result = []
    for symbol in POPULAR_STOCKS:
        if symbol in cache["stock_quotes"] and symbol in cache["company_profiles"]:
            quote = cache["stock_quotes"][symbol]
            profile = cache["company_profiles"][symbol]
            
            result.append({
                "symbol": symbol,
                "name": profile.get("name", ""),
                "price": quote.get("c", 0),
                "change": quote.get("d", 0),
                "changePercent": quote.get("dp", 0),
                "logo": profile.get("logo", "")
            })
    
    return result

@app.post("/api/analysis")
async def get_ai_analysis(symbol: str, company_name: str, price: float, change: float):
    """
    Hisse senedi için AI analizi döndürür.
    Önbellekte varsa ve son 24 saat içinde güncellendiyse, önbellekten döndürür.
    """
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
                'model': 'anthropic/claude-3-haiku',
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
        
        if response.status_code == 200:
            data = response.json()
            analysis = data['choices'][0]['message']['content']
            cache["ai_analysis"][cache_key] = analysis
            cache["last_updated"][f"ai_analysis_{cache_key}"] = datetime.now()
            return {"analysis": analysis}
        else:
            raise HTTPException(status_code=response.status_code, 
                               detail=f"Failed to get AI analysis: {response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            "stock_candles_count": len(cache["stock_candles"]),
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
