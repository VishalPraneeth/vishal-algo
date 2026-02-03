"""
Core Logic for Sector Heatmap & Market Strength Dashboard
Features:
1. Real-time sector heatmap with color coding
2. Opening Range Breakout (ORB) detection
3. Volume surge detection (2x 10-day average)
4. R-Factor calculation
5. Sectoral strength aggregation
6. Intelligent caching and rate limiting
"""

import asyncio
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import json
# from utils.logging import get_logger
import logging
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Data Structures
# ---------------------------------------------------

class MarketStrength(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class StockData:
    """Container for individual stock data"""
    symbol: str
    exchange: str
    ltp: float
    open: float
    high: float
    low: float
    volume: int
    previous_close: float
    timestamp: datetime
    average_volume: Optional[float] = None  # 10-day average
    opening_range_high: Optional[float] = None
    opening_range_low: Optional[float] = None
    
    @property
    def percentage_change(self) -> float:
        """Calculate percentage change from previous close"""
        if self.previous_close > 0:
            return ((self.ltp - self.previous_close) / self.previous_close) * 100
        return 0.0
    
    @property
    def r_factor(self) -> float:
        """Calculate R-Factor (Day's Range Position %)"""
        if self.high > self.low:
            return ((self.ltp - self.low) / (self.high - self.low)) * 100
        return 50.0  # Neutral if no movement
    
    @property
    def volume_surge_ratio(self) -> Optional[float]:
        """Calculate volume surge ratio vs 10-day average"""
        if self.average_volume and self.average_volume > 0:
            return self.volume / self.average_volume
        return None
    
    @property
    def is_orb_breakout(self) -> bool:
        """Check if price broke above opening range high"""
        if self.opening_range_high:
            return self.ltp > self.opening_range_high
        return False
    
    @property
    def orb_breakout_percent(self) -> Optional[float]:
        """Calculate ORB breakout percentage"""
        if self.is_orb_breakout and self.opening_range_high:
            return ((self.ltp - self.opening_range_high) / self.opening_range_high) * 100
        return None

@dataclass
class SectorMetrics:
    """Container for sector-level metrics"""
    sector_name: str
    stocks: List[StockData]
    avg_percentage_change: float
    avg_r_factor: float
    strength_score: float
    market_strength: MarketStrength
    
    @property
    def top_gainers(self, limit: int = 3) -> List[StockData]:
        """Get top gaining stocks in sector"""
        return sorted(self.stocks, key=lambda x: x.percentage_change, reverse=True)[:limit]
    
    @property
    def top_losers(self, limit: int = 3) -> List[StockData]:
        """Get top losing stocks in sector"""
        return sorted(self.stocks, key=lambda x: x.percentage_change)[:limit]

# ---------------------------------------------------
# Sector Mapper (Load from JSON)
# ---------------------------------------------------

class SectorMapper:
    """Manage sector-to-stock mapping"""
    
    def __init__(self, mapper_path: str = "../../data/sector_mapper.json"):
        self.mapper_path = mapper_path
        self.sector_map = self._load_mapper()
        self._reverse_map = self._build_reverse_map()
    
    def _load_mapper(self) -> Dict:
        """Load sector mapper from JSON file"""
        try:
            with open(self.mapper_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Sector mapper file not found: {self.mapper_path}")
            return self._get_default_mapper()
    
    def _get_default_mapper(self) -> Dict:
        """Return default sector mapper"""
        return {
            # 1) Core Financials
            "Private_Banks": [
                "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK",
                "INDUSINDBK", "AUBANK", "IDFCFIRSTB", "FEDERALBNK",
                "RBLBANK"
            ],
            "PSU_Banks": [
                "SBIN", "BANKBARODA", "PNB", "CANBK",
                "BANKINDIA", "UNIONBANK", "INDIANB"
            ],

            "Large_NBFCs": [
                "BAJFINANCE", "BAJAJFINSV", "SHRIRAMFIN",
                "CHOLAFIN", "MUTHOOTFIN", "LICHSGFIN",
                "PFC", "RECLTD", "JIOFIN", "MANAPPURAM", "IIFL"
            ],

            # 2) Insurance & AMC / Capital Markets
            "Insurance": [
                "SBILIFE", "HDFCLIFE", "ICICIPRULI", "LICI", "ICICIGI"
            ],
            "Exchanges_And_Brokers": [
                "BSE", "MCX", "ANGELONE", "NUVAMA"
            ],
            "Asset_Managers_AMC": [
                "HDFCAMC"
            ],
            "Cards_And_Payments": [
                "SBICARD", "PAYTM"  # payments/fintech
            ],

            # 3) IT & Tech
            "IT_Services": [
                "TCS", "INFY", "HCLTECH", "WIPRO", "LTIM",
                "TECHM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS",
                "TATAELXSI", "KPITTECH", "LTI"  # adjust as needed
            ],
            "Platform_And_Internet": [
                "NAUKRI", "NYKAA", "POLICYBZR", "ZOMATO",  # if/when F&O
                "DELHIVERY"
            ],

            # 4) Auto & Auto Ancillaries
            "Auto_OEM_4W_2W": [
                "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT",
                "TVSMOTOR", "HEROMOTOCO", "TMPV"
            ],
            "Auto_Ancillaries": [
                "MOTHERSON", "BHARATFORG", "BOSCHLTD",
                "SONACOMS", "UNOMINDA", "EXIDEIND", "TIINDIA"
            ],

            # 5) Metals & Mining
            "Metals_Private": [
                "ADANIENT", "JSWSTEEL", "TATASTEEL", "VEDL", "JINDALSTEL", "APLAPOLLO"
            ],
            "Metals_PSU_And_Mining": [
                "HINDALCO", "NMDC", "HINDZINC", "SAIL", "NATIONALUM"
            ],

            # 6) Oil, Gas & Energy
            "Oil_Gas_PSU": [
                "ONGC", "IOC", "BPCL", "GAIL", "OIL"
            ],
            "Oil_Gas_Private_CityGas": [
                "RELIANCE", "PETRONET"  # you dropped IGL/MGL as non-F&O
            ],
            "Power_Generation_And_Utilities": [
                "NTPC", "NHPC", "POWERGRID", "TATAPOWER",
                "TORNTPOWER", "ADANIGREEN"
            ],

            # 7) Consumption
            "FMCG_Staples": [
                "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA",
                "TATACONSUM", "VBL", "GODREJCP", "DABUR",
                "MARICO", "COLPAL"
            ],
            "Discretionary_And_Retail": [
                "TRENT", "DMART", "UNITDSPR", "JUBLFOOD",
                "PAGEIND", "UNITDSPR"  # beverages and retail
            ],

            # 8) Consumer Durables & Lifestyle
            "Consumer_Durables": [
                "TITAN", "HAVELLS", "DIXON", "KALYANKJIL",
                "VOLTAS", "BLUESTARCO", "AMBER", "CROMPTON",
                "PGEL"
            ],

            # 9) Healthcare & Pharma
            "Pharma_Large": [
                "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB",
                "TORNTPHARM", "ZYDUSLIFE", "MANKIND", "LUPIN",
                "AUROPHARMA", "ALKEM", "GLENMARK", "BIOCON",
                "LAURUSLABS"
            ],
            "Hospitals_And_Diagnostics": [
                "APOLLOHOSP", "MAXHEALTH", "FORTIS"
            ],

            # 10) Industrials, Infra & Capital Goods
            "Capital_Goods_And_Engineering": [
                "ABB", "SIEMENS", "CUMMINSIND", "BHEL", "BEL",
                "CGPOWER", "SOLARINDS", "KAYNES", "POWERINDIA"
            ],
            "Infra_And_Construction": [
                "LT", "RVNL", "NCC", "NBCC", "DLF", "LODHA",
                "GODREJPROP", "PRESTIGE", "OBEROIRLTY", "PHOENIXLTD"
            ],

            # 11) Materials & Cement
            "Cement_And_Building_Materials": [
                "ULTRACEMCO", "AMBUJACEM", "DALBHARAT", "SHREECEM",
                "ASTRAL", "POLYCAB", "PIDILITIND", "PIIND", "SRF"
            ],

            # 12) Telecom & Media
            "Telecom": [
                "BHARTIARTL", "IDEA"
            ],

            # 13) Travel, Hotels, Misc Services
            "Travel_Hotels_And_Leisure": [
                "INDIGO", "IRCTC", "INDHOTEL"
            ]
        }

    
    def _build_reverse_map(self) -> Dict:
        """Build symbol->sector reverse mapping"""
        reverse = {}
        for sector, symbols in self.sector_map.items():
            for symbol in symbols:
                reverse[symbol] = sector
        return reverse
    
    def get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol"""
        return self._reverse_map.get(symbol)
    
    def get_symbols_for_sector(self, sector: str) -> List[str]:
        """Get all symbols for a sector"""
        return self.sector_map.get(sector, [])
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols across all sectors"""
        all_symbols = []
        for symbols in self.sector_map.values():
            all_symbols.extend(symbols)
        return list(set(all_symbols))  # Remove duplicates if any

# ---------------------------------------------------
# Data Fetcher with Rate Limiting
# ---------------------------------------------------

class OpenAlgoDataFetcher:
    """Handle all OpenAlgo API calls with rate limiting"""
    
    def __init__(self, client, max_requests_per_sec: int = 3):
        self.client = client
        self.max_requests_per_sec = max_requests_per_sec
        self.last_request_time = 0
        self.request_interval = 1.0 / max_requests_per_sec
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
    async def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_interval:
            sleep_time = self.request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def fetch_multiquotes(self, symbols: List[str], exchange: str = "NSE") -> Dict:
        """Fetch multiple quotes with batch processing"""
        await self._rate_limit()
        
        # Prepare symbols for OpenAlgo API
        quote_symbols = [{"symbol": s, "exchange": exchange} for s in symbols]
        
        try:
            # Batch process (OpenAlgo may have limits per request)
            batch_size = 50
            all_results = []
            
            for i in range(0, len(quote_symbols), batch_size):
                batch = quote_symbols[i:i+batch_size]
                response = self.client.multiquotes(symbols=batch)
                
                if "results" in response:
                    all_results.extend(response["results"])
                
                # Respect rate limit between batches
                if i + batch_size < len(quote_symbols):
                    await asyncio.sleep(0.1)  # Small delay between batches
            
            return self._parse_multiquotes_response(all_results)
            
        except Exception as e:
            logger.error(f"Error fetching multiquotes: {e}")
            return {}
    
    def _parse_multiquotes_response(self, results: List) -> Dict[str, StockData]:
        """Parse OpenAlgo multiquotes response into StockData objects"""
        stock_data = {}
        
        for item in results:
            try:
                symbol = item["symbol"]
                data = item["data"]
                
                stock = StockData(
                    symbol=symbol,
                    exchange=item.get("exchange", "NSE"),
                    ltp=data.get("ltp", 0),
                    open=data.get("open", 0),
                    high=data.get("high", 0),
                    low=data.get("low", 0),
                    volume=data.get("volume", 0),
                    previous_close=data.get("prev_close", 0),
                    timestamp=datetime.now()
                )
                
                stock_data[symbol] = stock
                
            except KeyError as e:
                logger.warning(f"Missing key in response for {item.get('symbol', 'unknown')}: {e}")
        
        return stock_data
    
    async def fetch_historical_volume(self, symbol: str, exchange: str = "NSE", 
                                      days: int = 10) -> Optional[float]:
        """Fetch 10-day average volume for a symbol"""
        cache_key = f"{exchange}:{symbol}:volume_avg"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['value']
        
        await self._rate_limit()
        
        try:
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days+5)).strftime("%Y-%m-%d")
            
            response = self.client.history(
                symbol=symbol,
                exchange=exchange,
                interval="D",
                start_date=start_date,
                end_date=end_date
            )
            
            if isinstance(response, pd.DataFrame) and not response.empty:
                # Take last 'days' trading days for average
                recent_data = response.tail(days)
                if 'volume' in recent_data.columns:
                    avg_volume = recent_data['volume'].mean()
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'value': avg_volume,
                        'timestamp': time.time()
                    }
                    
                    return avg_volume
            
        except Exception as e:
            logger.error(f"Error fetching historical volume for {symbol}: {e}")
        
        return None
    
    async def fetch_opening_range(self, symbol: str, exchange: str = "NSE") -> Tuple[Optional[float], Optional[float]]:
        """Fetch opening range (9:15-9:30) high and low"""
        cache_key = f"{exchange}:{symbol}:opening_range"
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['high'], cached_data['low']
        
        await self._rate_limit()
        
        try:
            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Fetch 1-minute candles for opening range
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
                
                # Cache the result
                self.cache[cache_key] = {
                    'high': opening_high,
                    'low': opening_low,
                    'timestamp': time.time()
                }
                
                return opening_high, opening_low
            
        except Exception as e:
            logger.error(f"Error fetching opening range for {symbol}: {e}")
        
        return None, None

# ---------------------------------------------------
# Market Analysis Engine
# ---------------------------------------------------

class MarketAnalysisEngine:
    """Core analysis engine for market strength indicators"""
    
    def __init__(self, data_fetcher: OpenAlgoDataFetcher, sector_mapper: SectorMapper):
        self.data_fetcher = data_fetcher
        self.sector_mapper = sector_mapper
        self.stock_cache = {}
        self.sector_cache = {}
        self.indicators_cache = {}
        
    async def analyze_sector(self, sector: str, exchange: str = "NSE") -> SectorMetrics:
        """Analyze a specific sector"""
        cache_key = f"{exchange}:{sector}"
        
        # Check cache (5 minute TTL)
        if cache_key in self.sector_cache:
            cached_data = self.sector_cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 300:
                return cached_data['data']
        
        # Get symbols for sector
        symbols = self.sector_mapper.get_symbols_for_sector(sector)
        
        if not symbols:
            logger.warning(f"No symbols found for sector: {sector}")
            return None
        
        # Fetch live data for all symbols in sector
        stock_data = await self.data_fetcher.fetch_multiquotes(symbols, exchange)
        
        # Enhance stock data with additional metrics
        enhanced_stocks = []
        for symbol, stock in stock_data.items():
            enhanced_stock = await self._enhance_stock_data(stock)
            enhanced_stocks.append(enhanced_stock)
        
        # Calculate sector metrics
        sector_metrics = self._calculate_sector_metrics(sector, enhanced_stocks)
        
        # Cache the result
        self.sector_cache[cache_key] = {
            'data': sector_metrics,
            'timestamp': datetime.now()
        }
        
        return sector_metrics
    
    async def _enhance_stock_data(self, stock: StockData) -> StockData:
        """Enhance stock data with volume average and opening range"""
        # Get 10-day average volume (only if we haven't fetched today)
        if stock.average_volume is None:
            avg_volume = await self.data_fetcher.fetch_historical_volume(
                stock.symbol, stock.exchange
            )
            stock.average_volume = avg_volume
        
        # Get opening range (only during market hours)
        current_time = datetime.now().time()
        market_start = datetime.strptime("09:30", "%H:%M").time()
        
        if current_time > market_start and (stock.opening_range_high is None):
            opening_high, opening_low = await self.data_fetcher.fetch_opening_range(
                stock.symbol, stock.exchange
            )
            stock.opening_range_high = opening_high
            stock.opening_range_low = opening_low
        
        return stock
    
    def _calculate_sector_metrics(self, sector_name: str, stocks: List[StockData]) -> SectorMetrics:
        """Calculate aggregated sector metrics"""
        if not stocks:
            return None
        
        # Filter out invalid stocks
        valid_stocks = [s for s in stocks if s.previous_close > 0]
        
        if not valid_stocks:
            return None
        
        # Calculate averages
        avg_pct_change = np.mean([s.percentage_change for s in valid_stocks])
        avg_r_factor = np.mean([s.r_factor for s in valid_stocks])
        
        # Calculate strength score (weighted combination)
        # Higher percentage change and R-Factor → higher strength
        strength_score = (avg_pct_change * 0.6) + (avg_r_factor * 0.1)
        
        # Determine market strength category
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
        
        return SectorMetrics(
            sector_name=sector_name,
            stocks=valid_stocks,
            avg_percentage_change=avg_pct_change,
            avg_r_factor=avg_r_factor,
            strength_score=strength_score,
            market_strength=market_strength
        )
    
    async def detect_breakout_stocks(self, exchange: str = "NSE", 
                                     min_volume: int = 100000) -> List[Dict]:
        """Detect stocks with Opening Range Breakout and Volume Surge"""
        all_symbols = self.sector_mapper.get_all_symbols()
        breakout_stocks = []
        
        # Fetch data for all symbols
        stock_data = await self.data_fetcher.fetch_multiquotes(all_symbols, exchange)
        
        for symbol, stock in stock_data.items():
            # Skip low volume stocks
            if stock.volume < min_volume:
                continue
            
            # Enhance with additional data
            enhanced_stock = await self._enhance_stock_data(stock)
            
            # Check for ORB
            if enhanced_stock.is_orb_breakout:
                # Check for volume surge
                volume_surge = enhanced_stock.volume_surge_ratio
                has_volume_surge = volume_surge and volume_surge >= 2.0
                
                breakout_stocks.append({
                    'symbol': symbol,
                    'sector': self.sector_mapper.get_sector(symbol),
                    'ltp': enhanced_stock.ltp,
                    'orb_high': enhanced_stock.opening_range_high,
                    'orb_breakout_percent': enhanced_stock.orb_breakout_percent,
                    'r_factor': enhanced_stock.r_factor,
                    'volume': enhanced_stock.volume,
                    'volume_surge_ratio': volume_surge,
                    'has_volume_surge': has_volume_surge,
                    'percentage_change': enhanced_stock.percentage_change
                })
        
        # Sort by breakout percentage (strongest first)
        breakout_stocks.sort(key=lambda x: x.get('orb_breakout_percent', 0), reverse=True)
        
        return breakout_stocks[:20]  # Return top 20
    
    async def get_top_r_factor_stocks(self, exchange: str = "NSE", 
                                      limit: int = 10) -> List[Dict]:
        """Get stocks with highest R-Factor"""
        all_symbols = self.sector_mapper.get_all_symbols()
        r_factor_stocks = []
        
        stock_data = await self.data_fetcher.fetch_multiquotes(all_symbols, exchange)
        
        for symbol, stock in stock_data.items():
            r_factor_stocks.append({
                'symbol': symbol,
                'sector': self.sector_mapper.get_sector(symbol),
                'r_factor': stock.r_factor,
                'ltp': stock.ltp,
                'high': stock.high,
                'low': stock.low,
                'percentage_change': stock.percentage_change
            })
        
        # Sort by R-Factor (highest first)
        r_factor_stocks.sort(key=lambda x: x['r_factor'], reverse=True)
        
        return r_factor_stocks[:limit]
    
    async def analyze_all_sectors(self, exchange: str = "NSE") -> Dict:
        """Analyze all sectors and return comprehensive market view"""
        sectors = list(self.sector_mapper.sector_map.keys())
        
        # Analyze all sectors concurrently
        sector_tasks = []
        for sector in sectors:
            task = self.analyze_sector(sector, exchange)
            sector_tasks.append(task)
        
        sector_results = await asyncio.gather(*sector_tasks, return_exceptions=True)
        
        # Process results
        sector_metrics = {}
        for sector, result in zip(sectors, sector_results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing sector {sector}: {result}")
                continue
            if result:
                sector_metrics[sector] = result
        
        # Get additional market indicators
        breakout_stocks = await self.detect_breakout_stocks(exchange)
        top_r_factor_stocks = await self.get_top_r_factor_stocks(exchange)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'exchange': exchange,
            'sectors': sector_metrics,
            'breakout_stocks': breakout_stocks,
            'top_r_factor_stocks': top_r_factor_stocks,
            'market_summary': self._generate_market_summary(sector_metrics)
        }
    
    def _generate_market_summary(self, sector_metrics: Dict[str, SectorMetrics]) -> Dict:
        """Generate overall market summary"""
        if not sector_metrics:
            return {}
        
        # Count sector strengths
        strength_counts = defaultdict(int)
        total_sectors = len(sector_metrics)
        
        for sector, metrics in sector_metrics.items():
            strength_counts[metrics.market_strength.value] += 1
        
        # Calculate market breadth
        positive_sectors = sum(1 for m in sector_metrics.values() 
                             if m.avg_percentage_change > 0)
        negative_sectors = total_sectors - positive_sectors
        
        # Find strongest and weakest sectors
        sorted_sectors = sorted(sector_metrics.items(), 
                              key=lambda x: x[1].avg_percentage_change, 
                              reverse=True)
        
        strongest_sector = sorted_sectors[0] if sorted_sectors else None
        weakest_sector = sorted_sectors[-1] if sorted_sectors else None
        
        return {
            'total_sectors': total_sectors,
            'positive_sectors': positive_sectors,
            'negative_sectors': negative_sectors,
            'strength_distribution': dict(strength_counts),
            'strongest_sector': {
                'name': strongest_sector[0] if strongest_sector else None,
                'change': strongest_sector[1].avg_percentage_change if strongest_sector else None
            },
            'weakest_sector': {
                'name': weakest_sector[0] if weakest_sector else None,
                'change': weakest_sector[1].avg_percentage_change if weakest_sector else None
            }
        }

# ---------------------------------------------------
# Heatmap Generator
# ---------------------------------------------------

class HeatmapGenerator:
    """Generate heatmap visualizations"""
    
    @staticmethod
    def generate_sector_heatmap_data(sector_metrics: Dict[str, SectorMetrics]) -> pd.DataFrame:
        """Prepare data for sector heatmap visualization"""
        rows = []
        
        for sector_name, metrics in sector_metrics.items():
            rows.append({
                'Sector': sector_name,
                'Change %': metrics.avg_percentage_change,
                'R-Factor': metrics.avg_r_factor,
                'Strength Score': metrics.strength_score,
                'Market Strength': metrics.market_strength.value,
                'Stock Count': len(metrics.stocks),
                'Top Gainer': metrics.top_gainers[0].symbol if metrics.top_gainers else None,
                'Top Loser': metrics.top_losers[0].symbol if metrics.top_losers else None
            })
        
        df = pd.DataFrame(rows)
        
        # Add color coding
        def get_color_intensity(change):
            if change >= 2.0:
                return '#006400'  # Dark green
            elif change >= 0.5:
                return '#32CD32'  # Lime green
            elif change <= -2.0:
                return '#8B0000'  # Dark red
            elif change <= -0.5:
                return '#FF6347'  # Tomato red
            else:
                return '#D3D3D3'  # Light grey
        
        df['Color'] = df['Change %'].apply(get_color_intensity)
        
        return df
    
    @staticmethod
    def generate_stock_heatmap_data(stocks: List[StockData], sector: str = None) -> pd.DataFrame:
        """Prepare data for individual stock heatmap within a sector"""
        rows = []
        
        for stock in stocks:
            rows.append({
                'Symbol': stock.symbol,
                'Sector': sector or 'Unknown',
                'Change %': stock.percentage_change,
                'R-Factor': stock.r_factor,
                'Volume': stock.volume,
                'Volume Surge': stock.volume_surge_ratio or 0,
                'ORB Breakout': stock.is_orb_breakout,
                'LTP': stock.ltp,
                'High': stock.high,
                'Low': stock.low
            })
        
        return pd.DataFrame(rows)

# ---------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------

class MarketDashboardOrchestrator:
    """Main orchestrator for the market dashboard"""
    
    def __init__(self, openalgo_client, refresh_interval: int = 300):
        self.client = openalgo_client
        self.refresh_interval = refresh_interval  # seconds
        self.sector_mapper = SectorMapper()
        self.data_fetcher = OpenAlgoDataFetcher(openalgo_client)
        self.analysis_engine = MarketAnalysisEngine(self.data_fetcher, self.sector_mapper)
        self.heatmap_generator = HeatmapGenerator()
        self.is_running = False
        
    async def run_single_analysis(self, exchange: str = "NSE") -> Dict:
        """Run a single analysis cycle"""
        logger.info(f"Starting market analysis for {exchange}...")
        
        try:
            # Perform comprehensive analysis
            market_analysis = await self.analysis_engine.analyze_all_sectors(exchange)
            
            # Generate heatmap data
            sector_heatmap_df = self.heatmap_generator.generate_sector_heatmap_data(
                market_analysis['sectors']
            )
            
            # Prepare output
            output = {
                'analysis': market_analysis,
                'heatmap_data': sector_heatmap_df.to_dict('records'),
                'timestamp': datetime.now().isoformat(),
                'exchange': exchange
            }
            
            logger.info(f"Analysis completed at {output['timestamp']}")
            logger.info(f"Market Summary: {market_analysis['market_summary']}")
            
            # Print key findings
            self._print_key_findings(market_analysis)
            
            return output
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _print_key_findings(self, market_analysis: Dict):
        """Print key market findings to console"""
        print("\n" + "="*60)
        print("MARKET STRENGTH DASHBOARD - KEY FINDINGS")
        print("="*60)
        
        # Print sector performance
        print("\n📊 SECTOR PERFORMANCE:")
        print("-"*40)
        
        sectors = market_analysis.get('sectors', {})
        for sector_name, metrics in sorted(sectors.items(), 
                                         key=lambda x: x[1].avg_percentage_change, 
                                         reverse=True):
            change = metrics.avg_percentage_change
            strength = metrics.market_strength.value.replace('_', ' ').title()
            color = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
            
            print(f"{color} {sector_name:20} {change:>7.2f}% | {strength:15}")
        
        # Print breakout stocks
        print("\n🚀 BREAKOUT BEACON (ORB):")
        print("-"*40)
        
        breakout_stocks = market_analysis.get('breakout_stocks', [])
        for stock in breakout_stocks[:5]:  # Top 5
            if stock.get('orb_breakout_percent'):
                print(f"🔺 {stock['symbol']:10} | Sector: {stock['sector']:15} | "
                      f"Breakout: {stock['orb_breakout_percent']:.2f}% | "
                      f"Volume: {'🔥' if stock.get('has_volume_surge') else '📊'}")
        
        # Print R-Factor leaders
        print("\n🏆 R-FACTOR LEADERS:")
        print("-"*40)
        
        r_factor_stocks = market_analysis.get('top_r_factor_stocks', [])
        for stock in r_factor_stocks[:5]:  # Top 5
            print(f"🎯 {stock['symbol']:10} | Sector: {stock['sector']:15} | "
                  f"R-Factor: {stock['r_factor']:.1f}% | "
                  f"Change: {stock['percentage_change']:.2f}%")
        
        print("\n" + "="*60)
    
    async def run_continuous(self, exchange: str = "NSE"):
        """Run continuous analysis at specified intervals"""
        self.is_running = True
        
        logger.info(f"Starting continuous market analysis (Interval: {self.refresh_interval}s)")
        
        while self.is_running:
            try:
                await self.run_single_analysis(exchange)
                
                # Wait for next interval
                logger.info(f"Next analysis in {self.refresh_interval} seconds...")
                await asyncio.sleep(self.refresh_interval)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def stop(self):
        """Stop the continuous analysis"""
        self.is_running = False

# ---------------------------------------------------
# Usage Example
# ---------------------------------------------------

async def main():
    """Main function to demonstrate usage"""
    # Initialize OpenAlgo client (as per your example)
    from openalgo import api
    
    client = api(
        api_key="060f6506d732c6e2609d4c8dcaf612bcec9fc7bc92d49d05333a2ba544edcda1",
        host="http://127.0.0.1:5000"
    )
    
    print("🔁 Market Strength Dashboard - Core Logic")
    print("="*60)
    
    # Create orchestrator with 5-minute refresh
    orchestrator = MarketDashboardOrchestrator(client, refresh_interval=300)
    
    # Option 1: Run single analysis
    print("\nRunning single analysis...")
    result = await orchestrator.run_single_analysis("NSE")
    
    # Option 2: Run continuously (uncomment to use)
    # print("\nStarting continuous monitoring...")
    # await orchestrator.run_continuous("NSE")
    
    # You can also save results to files
    if 'analysis' in result:
        # Save sector data to CSV
        heatmap_df = pd.DataFrame(result['heatmap_data'])
        heatmap_df.to_csv('sector_heatmap.csv', index=False)
        print(f"\n✅ Sector heatmap saved to 'sector_heatmap.csv'")
        
        # Save breakout stocks to JSON
        breakout_stocks = result['analysis'].get('breakout_stocks', [])
        with open('breakout_stocks.json', 'w') as f:
            json.dump(breakout_stocks, f, indent=2)
        print(f"✅ Breakout stocks saved to 'breakout_stocks.json'")
        
        # Save full analysis to JSON
        with open('market_analysis.json', 'w') as f:
            # Convert dataclasses to dict
            def serialize(obj):
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)
            
            json.dump(result, f, default=serialize, indent=2)
        print(f"✅ Full analysis saved to 'market_analysis.json'")

# ---------------------------------------------------
# Fast API Integration (Optional - Future Enhancement)
# ---------------------------------------------------

"""
# This would be converted to API endpoints later
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    # Initialize orchestrator
    client = api(api_key="your_key", host="http://127.0.0.1:5000")
    orchestrator = MarketDashboardOrchestrator(client)

@app.get("/api/sector-heatmap")
async def get_sector_heatmap(exchange: str = "NSE"):
    result = await orchestrator.run_single_analysis(exchange)
    return JSONResponse(result)

@app.get("/api/breakout-stocks")
async def get_breakout_stocks(exchange: str = "NSE", limit: int = 10):
    stocks = await orchestrator.analysis_engine.detect_breakout_stocks(exchange)
    return JSONResponse(stocks[:limit])

@app.get("/api/r-factor-leaders")
async def get_r_factor_leaders(exchange: str = "NSE", limit: int = 10):
    stocks = await orchestrator.analysis_engine.get_top_r_factor_stocks(exchange, limit)
    return JSONResponse(stocks)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

# ---------------------------------------------------
# Entry Point
# ---------------------------------------------------

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())