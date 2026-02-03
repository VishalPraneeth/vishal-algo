import os
import time
from openalgo import api
import pandas as pd
import numpy as np
import datetime as datetime, time
# from data.sector_map import SECTOR_MAP, SECTOR_MAP_REFINED

SECTOR_MAP = {
  "IT": [
    "TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", 
    "TECHM", "PERSISTENT", "COFORGE", "MPHASIS", "OFSS"
  ],
  "Banking": [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", 
    "INDUSINDBK", "BANKBARODA", "PNB", "CANBK", "AUBANK", 
    "IDFCFIRSTB", "FEDERALBNK"
  ],
  "Auto": [
    "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT", 
    "TVSMOTOR", "HEROMOTOCO", "MOTHERSON", "BHARATFORG", "BOSCHLTD", 
    "ASHOKLEY", "TIINDIA", "SONACOMS", "UNOMINDA", "EXIDEIND"
  ],
  "Pharma": [
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "TORNTPHARM", 
    "ZYDUSLIFE", "MANKIND", "LUPIN", "AUROPHARMA", "ALKEM", 
    "ABBOTINDIA", "GLENMARK", "BIOCON", "IPCALAB", "AJANTPHARM", 
    "JBCHEPHARM", "GLAND", "WOCKPHARMA", "PIRPHARMA", "LAURUSLABS"
  ],
  "FMCG": [
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", 
    "VBL", "GODREJCP", "DABUR", "MARICO", "COLPAL", 
    "PGHH", "BALRAMCHIN", "UNITEDBN", "RADICO", "EMAMILTD"
  ],
  "Metal": [
    "ADANIENT", "JSWSTEEL", "TATASTEEL", "HINDALCO", "VEDL", 
    "JINDALSTEL", "NMDC", "HINDZINC", "SAIL", "NATIONALUM", 
    "JSL", "APLAPOLLO", "LLOYDSME", "HINDCOPPER", "WELCORP"
  ],
  "Financial_Services": [
    "HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "KOTAKBANK", 
    "AXISBANK", "BAJAJFINSV", "SBILIFE", "JIOFIN", "SHRIRAMFIN", 
    "HDFCLIFE", "CHOLAFIN", "MUTHOOTFIN", "PFC", "RECLTD", 
    "BSE", "ICICIGI", "ICICIPRULI", "SBICARD", "LICHSGFIN"
  ],
  "Oil_And_Gas": [
    "RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", 
    "HPCL", "OIL", "ATGL", "PETRONET", "IGL", 
    "GUJGASLTD", "MGL", "GSPL", "CASTROLIND", "AEGISCHEM"
  ],
  "Consumer_Durables": [
    "TITAN", "HAVELLS", "DIXON", "KALYANKJIL", "VOLTAS", 
    "BLUESTARCO", "AMBER", "CROMPTON", "VGUARD", "BATAINDIA", 
    "WHIRLPOOL", "KAJARIACER", "CENTURYPLY", "PGEL", "CERA"
  ],
  "Realty": [
    "DLF", "LODHA", "PRESTIGE", "GODREJPROP", "OBEROIRLTY", 
    "PHOENIXLTD", "BRIGADE", "SOBHA", "SIGNATURE", "ANANTRAJ"
  ]
}

RATE_LIMIT_SLEEP = 0.4  # 3 req/sec safe

api_key = '060f6506d732c6e2609d4c8dcaf612bcec9fc7bc92d49d05333a2ba544edcda1'

if not api_key:
    print("Error: OPENALGO_APIKEY environment variable not set")
    exit(1)

client = api(api_key, host='http://127.0.0.1:5000')

# --- 2. SCALABLE CACHE SYSTEM ---
# class MarketDataCache:
#     def __init__(self):
#         self.cache = {}
#         self.last_sync = None

#     def get_data(self, symbol, exchange="NSE"):
#         now = datetime.datetime.now()
#         # Cache logic: Valid for 5 minutes during market hours
#         if symbol in self.cache:
#             cache_entry = self.cache[symbol]
#             time_diff = (now - cache_entry['timestamp']).total_seconds()
#             if time_diff < 300:  # 5 minutes
#                 return cache_entry['data']

#         # Fetch New Data
#         try:
#             today_str = now.strftime("%Y-%m-%d")
#             # Fetching 5m candles to cover 9:15 to current
#             data = client.history(
#                 symbol=symbol, 
#                 exchange=exchange, 
#                 interval="5m", 
#                 start_date=today_str, 
#                 end_date=today_str
#             )
#             df = pd.DataFrame(data)
#             if not df.empty:
#                 self.cache[symbol] = {'timestamp': now, 'data': df}
#                 return df
#         except Exception as e:
#             print(f"Error fetching {symbol}: {e}")
#         return None

# market_cache = MarketDataCache()

# --- 3. ANALYTICS ENGINE ---
def calculate_metrics(df):
    """Calculates R-Factor and % Change from Day Start"""
    if df is None or df.empty:
        return None
    
    current_price = df['close'].iloc[-1]
    day_high = df['high'].max()
    day_low = df['low'].min()
    day_open = df['open'].iloc[0]
    
    # R-Factor Formula: (Current - Low) / (High - Low)
    # Handle zero division for flat stocks
    range_val = day_high - day_low
    r_factor = (current_price - day_low) / range_val if range_val > 0 else 0.5
    
    pct_change = ((current_price - day_open) / day_open) * 100
    
    return {
        "r_factor": round(r_factor * 100, 2), # Convert to %
        "pct_change": round(pct_change, 2),
        "last_price": current_price
    }

def classify_strength(avg_change):
    if avg_change > 0.75: return "STRONG_BUY"
    if avg_change > 0.25: return "BUY"
    if avg_change < -0.75: return "STRONG_SELL"
    if avg_change < -0.25: return "SELL"
    return "NEUTRAL"

# --- 4. EXECUTION LOOP ---
sector_results = []

# print(f"Starting Sectoral Intelligence Scan: {datetime.now().strftime('%H:%M:%S')}")

# for sector, symbols in SECTOR_MAP.items():
#     stock_metrics = []
    
#     for sym in symbols:
#         df = market_cache.get_data(sym)
#         metrics = calculate_metrics(df)
#         if metrics:
#             stock_metrics.append(metrics)
    
#     if stock_metrics:
#         avg_r = np.mean([m['r_factor'] for m in stock_metrics])
#         avg_pct = np.mean([m['pct_change'] for m in stock_metrics])
        
#         sector_results.append({
#             "Sector": sector,
#             "Avg_R_Factor_%": round(avg_r, 2),
#             "Avg_Pct_Change": round(avg_pct, 2),
#             "Signal": classify_strength(avg_pct),
#             "Constituents": len(symbols)
#         })

# Final Output
# intelligence_df = pd.DataFrame(sector_results).sort_values(by="Avg_R_Factor_%", ascending=False)
# display(intelligence_df)

def get_today_range():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return today, today

def fetch_sector_wise_scope(client, exchange="NSE", interval="5m"):
    start_date , end_date = get_today_range()
    sector_results = {}

    for sector , symbols in SECTOR_MAP.items():
        frames = []
        quote_symbols = [{"symbol": s, "exchange": "NSE"} for s in symbols]
        response = client.multiquotes(symbols=quote_symbols)
        print(symbols, type(response))

        # for symbol in symbols:
        #     try:
        #         # df = client.history(
        #         #     symbol=symbol,
        #         #     exchange=exchange,
        #         #     interval=interval,
        #         #     start_date="26-12-2025",
        #         #     end_date="26-12-2025"
        #         # )

        #         if not df.empty:
        #             df["pct_change"] = (
        #                 (df["close"] - df["open"]) / df["open"]
        #             ) * 100
        #             frames.append(df)

        #         time.sleep(RATE_LIMIT_SLEEP)

        #     except Exception as e:
        #         print(f"{symbol} error:", e)
        
        # if frames:
        #     combined = pd.concat(frames)
        #     sector_results[sector] = aggregate_sector(combined)

    return sector_results

def aggregate_sector(df):
    last_candle = df.iloc[-3:]  # last 3 x 5m candles

    avg_change = last_candle["pct_change"].mean()
    strength = classify_strength(avg_change)

    return {
        "avg_change": round(avg_change, 2),
        "strength": strength,
        "stocks": len(df["close"].unique())
    }

def classify_strength(avg_change):
    if avg_change > 0.75:
        return "STRONG_BUY"
    if avg_change > 0.25:
        return "BUY"
    if avg_change < -0.75:
        return "STRONG_SELL"
    if avg_change < -0.25:
        return "SELL"
    return "NEUTRAL"
