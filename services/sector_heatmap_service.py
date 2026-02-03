"""
Sector Heatmap API - Core logic for sector strength analysis
Optimized for speed with intelligent caching
"""

import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Data Structures
# ---------------------------------------------------

@dataclass
class StockData:
    symbol: str
    ltp: float
    open: float
    high: float
    low: float
    volume: int
    previous_close: float
    
    @property
    def percentage_change(self) -> float:
        if self.previous_close > 0:
            return ((self.ltp - self.previous_close) / self.previous_close) * 100
        return 0.0
    
    @property
    def r_factor(self) -> float:
        if self.high > self.low:
            return ((self.ltp - self.low) / (self.high - self.low)) * 100
        return 50.0

class MarketStrength(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class SectorMetrics:
    sector_name: str
    avg_percentage_change: float
    avg_r_factor: float
    strength_score: float
    market_strength: MarketStrength
    stock_count: int
    top_gainers: List[Dict]
    top_losers: List[Dict]

# ---------------------------------------------------
# Sector Mapper (Lightweight)
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
# Optimized Data Fetcher
# ---------------------------------------------------

class SectorDataFetcher:
    """Optimized for sector heatmap - only fetches essential data"""
    
    def __init__(self, client, max_requests_per_sec: int = 3):
        self.client = client
        self.max_requests_per_sec = max_requests_per_sec
        self.last_request_time = 0
        self.request_interval = 1.0 / max_requests_per_sec
        
        # Sector-level cache (5 minutes)
        self.sector_cache = {}
        self.cache_ttl = 300
    
    async def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def fetch_sector_data(self, exchange: str = "NSE") -> Dict[str, SectorMetrics]:
        """Fetch and process all sector data efficiently"""
        cache_key = f"sector_data_{exchange}"
        
        # Check cache
        if cache_key in self.sector_cache:
            cached = self.sector_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                logger.info("Returning cached sector data")
                return cached['data']
        
        logger.info(f"Fetching fresh sector data for {exchange}")
        
        # Initialize
        sector_mapper = SectorMapper()
        all_symbols = sector_mapper.get_all_symbols()
        
        # Fetch all stock data in minimal batches
        stock_data = await self._fetch_all_stock_data(all_symbols, exchange)
        
        # Process sectors
        sector_metrics = await self._process_sectors(sector_mapper, stock_data)
        
        # Cache results
        self.sector_cache[cache_key] = {
            'data': sector_metrics,
            'timestamp': time.time()
        }
        
        return sector_metrics
    
    async def _fetch_all_stock_data(self, symbols: List[str], exchange: str) -> Dict[str, StockData]:
        """Fetch all stock data with optimal batching"""
        await self._rate_limit()
        
        # Prepare symbols for batch request
        quote_symbols = [{"symbol": s, "exchange": exchange} for s in symbols]
        
        try:
            response = self.client.multiquotes(symbols=quote_symbols)
            
            if "results" not in response:
                logger.error("No results in API response")
                return {}
            
            # Parse response
            stock_data = {}
            for item in response["results"]:
                try:
                    symbol = item["symbol"]
                    data = item["data"]
                    
                    stock = StockData(
                        symbol=symbol,
                        ltp=data.get("ltp", 0),
                        open=data.get("open", 0),
                        high=data.get("high", 0),
                        low=data.get("low", 0),
                        volume=data.get("volume", 0),
                        previous_close=data.get("prev_close", 0)
                    )
                    
                    stock_data[symbol] = stock
                    
                except Exception as e:
                    logger.warning(f"Error parsing {item.get('symbol')}: {e}")
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return {}
    
    async def _process_sectors(self, sector_mapper: SectorMapper, 
                              stock_data: Dict[str, StockData]) -> Dict[str, SectorMetrics]:
        """Process stock data into sector metrics"""
        sector_stocks = defaultdict(list)
        
        # Group stocks by sector
        for symbol, stock in stock_data.items():
            sector = sector_mapper.get_sector(symbol)
            if sector:
                sector_stocks[sector].append(stock)
        
        # Calculate sector metrics
        sector_metrics = {}
        
        for sector_name, stocks in sector_stocks.items():
            if not stocks:
                continue
            
            # Filter valid stocks
            valid_stocks = [s for s in stocks if s.previous_close > 0]
            if not valid_stocks:
                continue
            
            # Calculate averages
            avg_pct_change = np.mean([s.percentage_change for s in valid_stocks])
            avg_r_factor = np.mean([s.r_factor for s in valid_stocks])
            
            # Calculate strength score
            strength_score = (avg_pct_change * 0.6) + (avg_r_factor * 0.1)
            
            # Determine market strength
            if avg_pct_change >= 2.0:
                market_strength = MarketStrength.STRONG_BULLISH
            elif avg_pct_change >= 0.5:
                market_strength = MarketStrength.BULLISH
            elif avg_pct_change <= -2.0:
                market_strength = MarketStrength.STRONG_BEARISH
            elif avg_pct_change <= -0.5:
                market_strength = MarketStrength.BEARISH
            else:
                market_strength = MarketStrength.NEUTRAL
            
            # Get top gainers and losers
            top_gainers = sorted(valid_stocks, 
                                key=lambda x: x.percentage_change, 
                                reverse=True)[:3]
            top_losers = sorted(valid_stocks, 
                               key=lambda x: x.percentage_change)[:3]
            
            sector_metrics[sector_name] = SectorMetrics(
                sector_name=sector_name,
                avg_percentage_change=avg_pct_change,
                avg_r_factor=avg_r_factor,
                strength_score=strength_score,
                market_strength=market_strength,
                stock_count=len(valid_stocks),
                top_gainers=[{
                    'symbol': s.symbol,
                    'change': s.percentage_change,
                    'ltp': s.ltp
                } for s in top_gainers],
                top_losers=[{
                    'symbol': s.symbol,
                    'change': s.percentage_change,
                    'ltp': s.ltp
                } for s in top_losers]
            )
        
        return sector_metrics

# ---------------------------------------------------
# Heatmap Generator
# ---------------------------------------------------

class HeatmapGenerator:
    @staticmethod
    def generate_heatmap_data(sector_metrics: Dict[str, SectorMetrics]) -> Dict:
        """Generate heatmap-ready data structure"""
        heatmap_data = []
        
        for sector_name, metrics in sector_metrics.items():
            # Determine color based on percentage change
            if metrics.avg_percentage_change >= 1.5:
                color = "#006400"  # Dark green
                intensity = 5
            elif metrics.avg_percentage_change >= 0.5:
                color = "#32CD32"  # Lime green
                intensity = 4
            elif metrics.avg_percentage_change > 0:
                color = "#90EE90"  # Light green
                intensity = 3
            elif metrics.avg_percentage_change >= -0.5:
                color = "#D3D3D3"  # Light grey
                intensity = 2
            elif metrics.avg_percentage_change >= -1.5:
                color = "#FF6347"  # Tomato red
                intensity = 1
            else:
                color = "#8B0000"  # Dark red
                intensity = 0
            
            heatmap_data.append({
                'sector': sector_name,
                'change_percent': round(metrics.avg_percentage_change, 2),
                'r_factor': round(metrics.avg_r_factor, 1),
                'strength_score': round(metrics.strength_score, 2),
                'market_strength': metrics.market_strength.value,
                'color': color,
                'intensity': intensity,
                'stock_count': metrics.stock_count,
                'top_gainers': metrics.top_gainers,
                'top_losers': metrics.top_losers
            })
        
        # Sort by change percent
        heatmap_data.sort(key=lambda x: x['change_percent'], reverse=True)
        
        return heatmap_data
    
    @staticmethod
    def generate_market_summary(sector_metrics: Dict[str, SectorMetrics]) -> Dict:
        """Generate overall market summary"""
        if not sector_metrics:
            return {}
        
        total_sectors = len(sector_metrics)
        positive_sectors = sum(1 for m in sector_metrics.values() 
                             if m.avg_percentage_change > 0)
        
        # Calculate breadth
        sector_changes = [m.avg_percentage_change for m in sector_metrics.values()]
        avg_market_change = np.mean(sector_changes) if sector_changes else 0
        
        # Find extremes
        sorted_sectors = sorted(sector_metrics.items(), 
                              key=lambda x: x[1].avg_percentage_change, 
                              reverse=True)
        
        strongest = sorted_sectors[0] if sorted_sectors else (None, None)
        weakest = sorted_sectors[-1] if sorted_sectors else (None, None)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_sectors': total_sectors,
            'positive_sectors': positive_sectors,
            'negative_sectors': total_sectors - positive_sectors,
            'avg_market_change': round(avg_market_change, 2),
            'market_breadth': f"{positive_sectors}/{total_sectors}",
            'strongest_sector': {
                'name': strongest[0],
                'change': round(strongest[1].avg_percentage_change, 2) if strongest[1] else None
            },
            'weakest_sector': {
                'name': weakest[0],
                'change': round(weakest[1].avg_percentage_change, 2) if weakest[1] else None
            }
        }

# ---------------------------------------------------
# Main Sector Heatmap API
# ---------------------------------------------------

class SectorHeatmapAPI:
    """Main API for sector heatmap data"""
    
    def __init__(self, openalgo_client):
        self.client = openalgo_client
        self.data_fetcher = SectorDataFetcher(openalgo_client)
        self.heatmap_generator = HeatmapGenerator()
    
    async def get_heatmap_data(self, exchange: str = "NSE") -> Dict:
        """Get complete heatmap data for API response"""
        start_time = time.time()
        
        try:
            # Fetch sector data
            sector_metrics = await self.data_fetcher.fetch_sector_data(exchange)
            
            if not sector_metrics:
                return {
                    'success': False,
                    'error': 'No sector data available',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Generate heatmap data
            heatmap_data = self.heatmap_generator.generate_heatmap_data(sector_metrics)
            market_summary = self.heatmap_generator.generate_market_summary(sector_metrics)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'exchange': exchange,
                'timestamp': datetime.now().isoformat(),
                'execution_time': round(execution_time, 2),
                'market_summary': market_summary,
                'sectors': heatmap_data,
                'sector_count': len(heatmap_data)
            }
            
        except Exception as e:
            logger.error(f"Error in get_heatmap_data: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_sector_details(self, sector_name: str, exchange: str = "NSE") -> Dict:
        """Get detailed data for a specific sector"""
        sector_metrics = await self.data_fetcher.fetch_sector_data(exchange)
        
        if sector_name not in sector_metrics:
            return {
                'success': False,
                'error': f'Sector {sector_name} not found',
                'timestamp': datetime.now().isoformat()
            }
        
        metrics = sector_metrics[sector_name]
        
        return {
            'success': True,
            'sector': sector_name,
            'exchange': exchange,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'avg_percentage_change': round(metrics.avg_percentage_change, 2),
                'avg_r_factor': round(metrics.avg_r_factor, 1),
                'strength_score': round(metrics.strength_score, 2),
                'market_strength': metrics.market_strength.value,
                'stock_count': metrics.stock_count
            },
            'top_gainers': metrics.top_gainers,
            'top_losers': metrics.top_losers
        }

# ---------------------------------------------------
# FastAPI Integration
# ---------------------------------------------------

"""
# To use with FastAPI:

from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()
sector_api = None

@app.on_event("startup")
async def startup():
    from openalgo import api
    client = api(api_key="your_key", host="http://127.0.0.1:5000")
    global sector_api
    sector_api = SectorHeatmapAPI(client)

@app.get("/api/sector-heatmap")
async def get_sector_heatmap(exchange: str = "NSE"):
    result = await sector_api.get_heatmap_data(exchange)
    if not result['success']:
        raise HTTPException(status_code=500, detail=result['error'])
    return result

@app.get("/api/sector/{sector_name}")
async def get_sector(sector_name: str, exchange: str = "NSE"):
    result = await sector_api.get_sector_details(sector_name, exchange)
    if not result['success']:
        raise HTTPException(status_code=404, detail=result['error'])
    return result
"""

# ---------------------------------------------------
# Command Line Interface
# ---------------------------------------------------

async def main():
    """CLI for testing sector heatmap"""
    from openalgo import api
    
    # Initialize client
    client = api(
        api_key="060f6506d732c6e2609d4c8dcaf612bcec9fc7bc92d49d05333a2ba544edcda1",
        host="http://127.0.0.1:5000"
    )
    
    print("🔥 Sector Heatmap API - Optimized")
    print("="*60)
    
    # Create API instance
    sector_api = SectorHeatmapAPI(client)
    
    # Get heatmap data
    print("\n📊 Fetching sector heatmap data...")
    start_time = time.time()
    
    result = await sector_api.get_heatmap_data("NSE")
    
    execution_time = time.time() - start_time
    print(f"✅ Execution time: {execution_time:.2f} seconds")
    
    if result['success']:
        # Display results
        print(f"\n📈 Market Summary:")
        print(f"   Sectors: {result['market_summary']['market_breadth']} positive")
        print(f"   Avg Change: {result['market_summary']['avg_market_change']}%")
        print(f"   Strongest: {result['market_summary']['strongest_sector']['name']} "
              f"({result['market_summary']['strongest_sector']['change']}%)")
        
        print("\n🎨 Sector Performance:")
        for sector in result['sectors'][:5]:  # Show top 5
            color_indicator = "🟢" if sector['change_percent'] > 0 else "🔴"
            print(f"{color_indicator} {sector['sector']:20} {sector['change_percent']:>7.2f}%")
        
        # Save to JSON
        with open('sector_heatmap_output.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Saved to 'sector_heatmap_output.json'")
        
        # Generate CSV for heatmap
        df = pd.DataFrame(result['sectors'])
        df.to_csv('sector_heatmap.csv', index=False)
        print(f"📊 CSV saved to 'sector_heatmap.csv'")
        
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())