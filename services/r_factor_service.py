"""
R-Factor Leaders API - Identify strongest momentum stocks
Lightweight and fast execution
"""

import asyncio
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from flask import json
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Data Structures
# ---------------------------------------------------

@dataclass
class RFactorStock:
    symbol: str
    sector: str
    r_factor: float
    ltp: float
    high: float
    low: float
    percentage_change: float
    volume: int
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
# R-Factor Calculator
# ---------------------------------------------------

class RFactorCalculator:
    """Calculate R-Factor for stocks efficiently"""
    
    def __init__(self, client, max_requests_per_sec: int = 3):
        self.client = client
        self.max_requests_per_sec = max_requests_per_sec
        self.last_request_time = 0
        self.request_interval = 1.0 / max_requests_per_sec
        
        # Cache
        self.r_factor_cache = {}
        self.cache_ttl = 60  # 1 minute cache
    
    async def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def calculate_r_factors(self, exchange: str = "NSE", 
                                 min_volume: int = 50000) -> List[RFactorStock]:
        """Calculate R-Factors for all stocks"""
        cache_key = f"r_factors_{exchange}"
        
        # Check cache
        if cache_key in self.r_factor_cache:
            cached = self.r_factor_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                logger.info("Returning cached R-Factors")
                return cached['data']
        
        logger.info("Calculating fresh R-Factors...")
        
        sector_mapper = SectorMapper()
        all_symbols = sector_mapper.get_all_symbols()
        
        await self._rate_limit()
        
        try:
            # Fetch all data in one batch
            quote_symbols = [{"symbol": s, "exchange": exchange} for s in all_symbols]
            response = self.client.multiquotes(symbols=quote_symbols)
            
            if "results" not in response:
                return []
            
            r_factor_stocks = []
            
            for item in response["results"]:
                try:
                    symbol = item["symbol"]
                    data = item["data"]
                    
                    ltp = data.get("ltp", 0)
                    high = data.get("high", 0)
                    low = data.get("low", 0)
                    volume = data.get("volume", 0)
                    prev_close = data.get("prev_close", 0)
                    open_price = data.get("open", 0)
                    
                    # Skip if invalid data
                    if prev_close <= 0 or volume < min_volume:
                        continue
                    
                    # Calculate R-Factor
                    if high > low:
                        r_factor = ((ltp - low) / (high - low)) * 100
                    else:
                        r_factor = 50.0
                    
                    # Calculate percentage change
                    pct_change = ((ltp - prev_close) / prev_close) * 100
                    
                    # Get sector
                    sector = sector_mapper.get_sector(symbol)
                    if not sector:
                        continue
                    
                    r_factor_stock = RFactorStock(
                        symbol=symbol,
                        sector=sector,
                        r_factor=r_factor,
                        ltp=ltp,
                        high=high,
                        low=low,
                        percentage_change=pct_change,
                        volume=volume,
                        timestamp=datetime.now()
                    )
                    
                    r_factor_stocks.append(r_factor_stock)
                    
                except Exception as e:
                    logger.warning(f"Error processing {item.get('symbol')}: {e}")
                    continue
            
            # Sort by R-Factor (highest first)
            r_factor_stocks.sort(key=lambda x: x.r_factor, reverse=True)
            
            # Cache results
            self.r_factor_cache[cache_key] = {
                'data': r_factor_stocks,
                'timestamp': time.time()
            }
            
            return r_factor_stocks
            
        except Exception as e:
            logger.error(f"Error calculating R-Factors: {e}")
            return []
    
    async def get_sector_r_factors(self, sector: str, exchange: str = "NSE") -> List[RFactorStock]:
        """Get R-Factors for a specific sector"""
        all_stocks = await self.calculate_r_factors(exchange)
        
        # Filter by sector
        sector_stocks = [s for s in all_stocks if s.sector == sector]
        sector_stocks.sort(key=lambda x: x.r_factor, reverse=True)
        
        return sector_stocks

# ---------------------------------------------------
# R-Factor Leaders API
# ---------------------------------------------------

class RFactorLeadersAPI:
    """Main API for R-Factor leaders"""
    
    def __init__(self, openalgo_client):
        self.client = openalgo_client
        self.calculator = RFactorCalculator(openalgo_client)
    
    async def get_r_factor_leaders(self, exchange: str = "NSE", 
                                  limit: int = 20,
                                  min_r_factor: float = 70.0) -> Dict:
        """Get top R-Factor leaders"""
        start_time = time.time()
        
        try:
            # Calculate R-Factors
            r_factor_stocks = await self.calculator.calculate_r_factors(exchange)
            
            # Apply filters
            filtered_stocks = [s for s in r_factor_stocks if s.r_factor >= min_r_factor]
            
            # Apply limit
            leaders = filtered_stocks[:limit]
            
            # Convert to dict for JSON serialization
            stocks_data = []
            for stock in leaders:
                # Determine strength level
                if stock.r_factor >= 90:
                    strength = "extremely_strong"
                elif stock.r_factor >= 80:
                    strength = "very_strong"
                elif stock.r_factor >= 70:
                    strength = "strong"
                else:
                    strength = "moderate"
                
                stocks_data.append({
                    'symbol': stock.symbol,
                    'sector': stock.sector,
                    'r_factor': round(stock.r_factor, 1),
                    'strength_level': strength,
                    'ltp': round(stock.ltp, 2),
                    'high': round(stock.high, 2),
                    'low': round(stock.low, 2),
                    'percentage_change': round(stock.percentage_change, 2),
                    'volume': stock.volume,
                    'timestamp': stock.timestamp.isoformat()
                })
            
            execution_time = time.time() - start_time
            
            # Calculate statistics
            total_stocks = len(r_factor_stocks)
            strong_stocks = len(filtered_stocks)  # Above min_r_factor
            
            # Sector distribution
            sector_distribution = {}
            for stock in leaders:
                sector = stock.sector
                sector_distribution[sector] = sector_distribution.get(sector, 0) + 1
            
            return {
                'success': True,
                'exchange': exchange,
                'timestamp': datetime.now().isoformat(),
                'execution_time': round(execution_time, 2),
                'filters': {
                    'limit': limit,
                    'min_r_factor': min_r_factor
                },
                'statistics': {
                    'total_stocks_analyzed': total_stocks,
                    'strong_stocks': strong_stocks,
                    'strong_ratio': f"{strong_stocks}/{total_stocks}",
                    'sector_distribution': sector_distribution
                },
                'leaders': stocks_data
            }
            
        except Exception as e:
            logger.error(f"Error in get_r_factor_leaders: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_sector_leaders(self, sector: str, exchange: str = "NSE", 
                                limit: int = 10) -> Dict:
        """Get R-Factor leaders for a specific sector"""
        try:
            sector_stocks = await self.calculator.get_sector_r_factors(sector, exchange)
            
            # Apply limit
            leaders = sector_stocks[:limit]
            
            # Convert to dict
            stocks_data = []
            for stock in leaders:
                stocks_data.append({
                    'symbol': stock.symbol,
                    'r_factor': round(stock.r_factor, 1),
                    'ltp': round(stock.ltp, 2),
                    'percentage_change': round(stock.percentage_change, 2),
                    'volume': stock.volume,
                    'timestamp': stock.timestamp.isoformat()
                })
            
            return {
                'success': True,
                'sector': sector,
                'exchange': exchange,
                'timestamp': datetime.now().isoformat(),
                'statistics': {
                    'total_in_sector': len(sector_stocks),
                    'avg_r_factor': round(np.mean([s.r_factor for s in sector_stocks]), 1) 
                                   if sector_stocks else 0,
                    'sector_strength': self._get_sector_strength_level(sector_stocks)
                },
                'leaders': stocks_data
            }
            
        except Exception as e:
            logger.error(f"Error in get_sector_leaders: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_sector_strength_level(self, stocks: List[RFactorStock]) -> str:
        """Determine sector strength based on average R-Factor"""
        if not stocks:
            return "neutral"
        
        avg_r_factor = np.mean([s.r_factor for s in stocks])
        
        if avg_r_factor >= 80:
            return "extremely_strong"
        elif avg_r_factor >= 70:
            return "very_strong"
        elif avg_r_factor >= 60:
            return "strong"
        elif avg_r_factor >= 40:
            return "moderate"
        else:
            return "weak"

# ---------------------------------------------------
# Command Line Interface
# ---------------------------------------------------

async def main():
    """CLI for testing R-Factor leaders"""
    from openalgo import api
    
    # Initialize client
    client = api(
        api_key="060f6506d732c6e2609d4c8dcaf612bcec9fc7bc92d49d05333a2ba544edcda1",
        host="http://127.0.0.1:5000"
    )
    
    print("🏆 R-Factor Leaders API - Optimized")
    print("="*60)
    
    # Create API instance
    rfactor_api = RFactorLeadersAPI(client)
    
    # Get R-Factor leaders
    print("\n🎯 Calculating R-Factor leaders...")
    start_time = time.time()
    
    result = await rfactor_api.get_r_factor_leaders("NSE", limit=15, min_r_factor=70)
    
    execution_time = time.time() - start_time
    print(f"✅ Execution time: {execution_time:.2f} seconds")
    
    if result['success']:
        print(f"\n📊 R-Factor Statistics:")
        print(f"   Stocks Analyzed: {result['statistics']['total_stocks_analyzed']}")
        print(f"   Strong Stocks (R ≥ 70): {result['statistics']['strong_ratio']}")
        
        print(f"\n📍 Sector Distribution:")
        for sector, count in result['statistics']['sector_distribution'].items():
            print(f"   {sector:20}: {count:2} stocks")
        
        if result['leaders']:
            print("\n🏆 Top R-Factor Leaders:")
            for i, stock in enumerate(result['leaders'][:10], 1):
                strength_emoji = "🔥" if stock['r_factor'] >= 90 else "⭐" if stock['r_factor'] >= 80 else "⚡"
                print(f"{i:2}. {stock['symbol']:10} | "
                      f"Sector: {stock['sector']:15} | "
                      f"R-Factor: {stock['r_factor']:>5.1f}% {strength_emoji} | "
                      f"Change: {stock['percentage_change']:>6.2f}%")
        else:
            print("\n📭 No strong R-Factor leaders found")
        
        # Save to JSON
        with open('r_factor_leaders_output.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Saved to 'r_factor_leaders_output.json'")
        
        # Get sector-specific leaders example
        print(f"\n🔍 Example: IT Sector Leaders")
        sector_result = await rfactor_api.get_sector_leaders("IT", "NSE", limit=5)
        
        if sector_result['success'] and sector_result['leaders']:
            for stock in sector_result['leaders']:
                print(f"   {stock['symbol']}: R={stock['r_factor']}%, Change={stock['percentage_change']}%")
        
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())