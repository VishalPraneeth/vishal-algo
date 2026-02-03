"""
Breakout Stocks API - Detect ORB and Volume Surge stocks
Optimized with selective data fetching
"""

import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from flask import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Data Structures
# ---------------------------------------------------

@dataclass
class BreakoutStock:
    symbol: str
    sector: str
    ltp: float
    opening_range_high: float
    orb_breakout_percent: float
    r_factor: float
    volume: int
    avg_volume: float
    volume_surge_ratio: float
    percentage_change: float
    has_volume_surge: bool
    timestamp: datetime

# ---------------------------------------------------
# Sector Mapper (Shared)
# ---------------------------------------------------

class SectorMapper:
    def __init__(self):
        self.sector_map = {
            "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM"],
            "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
            "Auto": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT"],
            "Pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "TORNTPHARM"],
            "Consumer_Durables": ["TITAN", "HAVELLS", "DIXON", "VOLTAS", "CROMPTON"],
            "Realty": ["DLF", "LODHA", "PRESTIGE", "GODREJPROP", "OBEROIRLTY"]
        }
        self._reverse_map = {}
        self._build_reverse_map()
    
    def _build_reverse_map(self):
        for sector, symbols in self.sector_map.items():
            for symbol in symbols:
                self._reverse_map[symbol] = sector
    
    def get_sector(self, symbol: str) -> Optional[str]:
        return self._reverse_map.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        all_symbols = []
        for symbols in self.sector_map.values():
            all_symbols.extend(symbols)
        return list(set(all_symbols))

# ---------------------------------------------------
# Intelligent Breakout Detector
# ---------------------------------------------------

class BreakoutDetector:
    """Detect ORB and Volume Surge with selective fetching"""
    
    def __init__(self, client, max_requests_per_sec: int = 3):
        self.client = client
        self.max_requests_per_sec = max_requests_per_sec
        self.last_request_time = 0
        self.request_interval = 1.0 / max_requests_per_sec
        
        # Caches
        self.historical_volume_cache = {}
        self.opening_range_cache = {}
        self.breakout_cache = {}
        self.cache_ttl = 300
    
    async def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def get_potential_breakout_candidates(self, exchange: str = "NSE") -> List[str]:
        """Get stocks showing early signs of breakout (pre-filter)"""
        await self._rate_limit()
        
        sector_mapper = SectorMapper()
        all_symbols = sector_mapper.get_all_symbols()
        
        # First, get basic data for all stocks
        quote_symbols = [{"symbol": s, "exchange": exchange} for s in all_symbols]
        
        try:
            response = self.client.multiquotes(symbols=quote_symbols)
            
            if "results" not in response:
                return []
            
            # Filter for stocks showing strength
            candidates = []
            for item in response["results"]:
                try:
                    symbol = item["symbol"]
                    data = item["data"]
                    
                    ltp = data.get("ltp", 0)
                    open_price = data.get("open", 0)
                    prev_close = data.get("prev_close", 0)
                    volume = data.get("volume", 0)
                    
                    # Basic filters
                    if prev_close <= 0:
                        continue
                    
                    # Rule 1: Already above open
                    if ltp <= open_price:
                        continue
                    
                    # Rule 2: Volume above threshold
                    if volume < 100000:  # Minimum volume
                        continue
                    
                    # Rule 3: Positive change
                    pct_change = ((ltp - prev_close) / prev_close) * 100
                    if pct_change < 0:
                        continue
                    
                    candidates.append(symbol)
                    
                except Exception:
                    continue
            
            logger.info(f"Found {len(candidates)} potential breakout candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
            return []
    
    async def _get_historical_volume(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """Get 10-day average volume with caching"""
        cache_key = f"{exchange}:{symbol}:volume"
        
        if cache_key in self.historical_volume_cache:
            cached = self.historical_volume_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['value']
        
        await self._rate_limit()
        
        try:
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
            
            response = self.client.history(
                symbol=symbol,
                exchange=exchange,
                interval="D",
                start_date=start_date,
                end_date=end_date
            )
            
            if isinstance(response, pd.DataFrame) and not response.empty:
                # Take last 10 trading days
                recent_data = response.tail(10)
                if 'volume' in recent_data.columns:
                    avg_volume = recent_data['volume'].mean()
                    
                    self.historical_volume_cache[cache_key] = {
                        'value': avg_volume,
                        'timestamp': time.time()
                    }
                    
                    return avg_volume
            
        except Exception as e:
            logger.warning(f"Error fetching historical volume for {symbol}: {e}")
        
        return None
    
    async def _get_opening_range(self, symbol: str, exchange: str = "NSE") -> Tuple[Optional[float], Optional[float]]:
        """Get opening range with caching"""
        cache_key = f"{exchange}:{symbol}:opening_range"
        
        if cache_key in self.opening_range_cache:
            cached = self.opening_range_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['high'], cached['low']
        
        await self._rate_limit()
        
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            response = self.client.history(
                symbol=symbol,
                exchange=exchange,
                interval="1minute",
                start_date=f"{today} 09:15:00",
                end_date=f"{today} 09:30:00"
            )
            
            if isinstance(response, pd.DataFrame) and not response.empty:
                opening_high = response['high'].max()
                opening_low = response['low'].min()
                
                self.opening_range_cache[cache_key] = {
                    'high': opening_high,
                    'low': opening_low,
                    'timestamp': time.time()
                }
                
                return opening_high, opening_low
            
        except Exception as e:
            logger.warning(f"Error fetching opening range for {symbol}: {e}")
        
        return None, None
    
    async def detect_breakouts(self, exchange: str = "NSE", 
                              min_breakout_percent: float = 0.5,
                              min_volume_surge: float = 1.5) -> List[BreakoutStock]:
        """Main detection logic"""
        cache_key = f"breakouts_{exchange}"
        
        # Check cache
        if cache_key in self.breakout_cache:
            cached = self.breakout_cache[cache_key]
            if time.time() - cached['timestamp'] < 60:  # 1 minute cache for breakouts
                logger.info("Returning cached breakouts")
                return cached['data']
        
        logger.info("Detecting fresh breakouts...")
        
        sector_mapper = SectorMapper()
        
        # Step 1: Get potential candidates
        candidates = await self.get_potential_breakout_candidates(exchange)
        
        if not candidates:
            return []
        
        # Step 2: Fetch detailed data for candidates only
        breakout_stocks = []
        
        for symbol in candidates:
            try:
                # Get sector
                sector = sector_mapper.get_sector(symbol)
                if not sector:
                    continue
                
                # Get current quote
                await self._rate_limit()
                quote_response = self.client.multiquotes(
                    symbols=[{"symbol": symbol, "exchange": exchange}]
                )
                
                if not quote_response.get("results"):
                    continue
                
                quote_data = quote_response["results"][0]["data"]
                ltp = quote_data.get("ltp", 0)
                high = quote_data.get("high", 0)
                low = quote_data.get("low", 0)
                volume = quote_data.get("volume", 0)
                prev_close = quote_data.get("prev_close", 0)
                
                # Skip if invalid data
                if prev_close <= 0:
                    continue
                
                # Calculate R-Factor
                if high > low:
                    r_factor = ((ltp - low) / (high - low)) * 100
                else:
                    r_factor = 50.0
                
                # Get opening range (async)
                opening_high, opening_low = await self._get_opening_range(symbol, exchange)
                
                # Check ORB
                if opening_high and ltp > opening_high:
                    orb_breakout_percent = ((ltp - opening_high) / opening_high) * 100
                    
                    # Filter by minimum breakout percent
                    if orb_breakout_percent >= min_breakout_percent:
                        # Get historical volume (async)
                        avg_volume = await self._get_historical_volume(symbol, exchange)
                        
                        if avg_volume and avg_volume > 0:
                            volume_surge_ratio = volume / avg_volume
                            has_volume_surge = volume_surge_ratio >= min_volume_surge
                            
                            # Calculate percentage change
                            pct_change = ((ltp - prev_close) / prev_close) * 100
                            
                            breakout_stock = BreakoutStock(
                                symbol=symbol,
                                sector=sector,
                                ltp=ltp,
                                opening_range_high=opening_high,
                                orb_breakout_percent=orb_breakout_percent,
                                r_factor=r_factor,
                                volume=volume,
                                avg_volume=avg_volume,
                                volume_surge_ratio=volume_surge_ratio,
                                percentage_change=pct_change,
                                has_volume_surge=has_volume_surge,
                                timestamp=datetime.now()
                            )
                            
                            breakout_stocks.append(breakout_stock)
                
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by breakout percentage (strongest first)
        breakout_stocks.sort(key=lambda x: x.orb_breakout_percent, reverse=True)
        
        # Cache results
        self.breakout_cache[cache_key] = {
            'data': breakout_stocks,
            'timestamp': time.time()
        }
        
        return breakout_stocks

# ---------------------------------------------------
# Breakout Stocks API
# ---------------------------------------------------

class BreakoutStocksAPI:
    """Main API for breakout stocks"""
    
    def __init__(self, openalgo_client):
        self.client = openalgo_client
        self.detector = BreakoutDetector(openalgo_client)
    
    async def get_breakout_stocks(self, exchange: str = "NSE", 
                                 min_breakout_percent: float = 0.5,
                                 min_volume_surge: float = 1.5,
                                 limit: int = 20) -> Dict:
        """Get breakout stocks with filters"""
        start_time = time.time()
        
        try:
            # Detect breakouts
            breakout_stocks = await self.detector.detect_breakouts(
                exchange, min_breakout_percent, min_volume_surge
            )
            
            # Apply limit
            breakout_stocks = breakout_stocks[:limit]
            
            # Convert to dict for JSON serialization
            stocks_data = []
            for stock in breakout_stocks:
                stocks_data.append({
                    'symbol': stock.symbol,
                    'sector': stock.sector,
                    'ltp': round(stock.ltp, 2),
                    'opening_range_high': round(stock.opening_range_high, 2),
                    'orb_breakout_percent': round(stock.orb_breakout_percent, 2),
                    'r_factor': round(stock.r_factor, 1),
                    'volume': stock.volume,
                    'avg_volume': round(stock.avg_volume, 2),
                    'volume_surge_ratio': round(stock.volume_surge_ratio, 2),
                    'percentage_change': round(stock.percentage_change, 2),
                    'has_volume_surge': stock.has_volume_surge,
                    'timestamp': stock.timestamp.isoformat()
                })
            
            execution_time = time.time() - start_time
            
            # Calculate statistics
            total_detected = len(breakout_stocks)
            with_volume_surge = sum(1 for s in breakout_stocks if s.has_volume_surge)
            
            return {
                'success': True,
                'exchange': exchange,
                'timestamp': datetime.now().isoformat(),
                'execution_time': round(execution_time, 2),
                'filters': {
                    'min_breakout_percent': min_breakout_percent,
                    'min_volume_surge': min_volume_surge,
                    'limit': limit
                },
                'statistics': {
                    'total_breakouts': total_detected,
                    'with_volume_surge': with_volume_surge,
                    'breakout_ratio': f"{with_volume_surge}/{total_detected}"
                },
                'stocks': stocks_data
            }
            
        except Exception as e:
            logger.error(f"Error in get_breakout_stocks: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_strong_breakouts(self, exchange: str = "NSE") -> Dict:
        """Get only strong breakouts (ORB > 1% with volume surge)"""
        return await self.get_breakout_stocks(
            exchange=exchange,
            min_breakout_percent=1.0,
            min_volume_surge=2.0,
            limit=10
        )
    
    async def get_sector_breakouts(self, sector: str, exchange: str = "NSE") -> Dict:
        """Get breakouts for a specific sector"""
        result = await self.get_breakout_stocks(exchange=exchange)
        
        if not result['success']:
            return result
        
        # Filter by sector
        sector_stocks = [s for s in result['stocks'] if s['sector'] == sector]
        
        result['stocks'] = sector_stocks
        result['statistics']['sector_breakouts'] = len(sector_stocks)
        
        return result

# ---------------------------------------------------
# Command Line Interface
# ---------------------------------------------------

async def main():
    """CLI for testing breakout detection"""
    from openalgo import api
    
    # Initialize client
    client = api(
        api_key="060f6506d732c6e2609d4c8dcaf612bcec9fc7bc92d49d05333a2ba544edcda1",
        host="http://127.0.0.1:5000"
    )
    
    print("🚀 Breakout Stocks API - Optimized")
    print("="*60)
    
    # Create API instance
    breakout_api = BreakoutStocksAPI(client)
    
    # Get breakout stocks
    print("\n🔍 Detecting breakout stocks...")
    start_time = time.time()
    
    result = await breakout_api.get_breakout_stocks("NSE")
    
    execution_time = time.time() - start_time
    print(f"✅ Execution time: {execution_time:.2f} seconds")
    
    if result['success']:
        print(f"\n📊 Breakout Statistics:")
        print(f"   Total Breakouts: {result['statistics']['total_breakouts']}")
        print(f"   With Volume Surge: {result['statistics']['with_volume_surge']}")
        
        if result['stocks']:
            print("\n🔥 Top Breakout Stocks:")
            for i, stock in enumerate(result['stocks'][:10], 1):
                volume_indicator = "🔥" if stock['has_volume_surge'] else "📊"
                print(f"{i:2}. {stock['symbol']:10} | "
                      f"Sector: {stock['sector']:15} | "
                      f"ORB: {stock['orb_breakout_percent']:>5.2f}% | "
                      f"R-Factor: {stock['r_factor']:>4.1f}% | "
                      f"Volume: {volume_indicator} {stock['volume_surge_ratio']:.1f}x")
        else:
            print("\n📭 No breakout stocks detected")
        
        # Save to JSON
        with open('breakout_stocks_output.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Saved to 'breakout_stocks_output.json'")
        
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())