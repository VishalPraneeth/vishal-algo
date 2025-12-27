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
# Keep adding new additions in the sectors or include more sectors

SECTOR_MAP_REFINED = {
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

# relative strength data set
{
  "meta": {
    "version": "2025.12",
    "units": "percent",
    "timeframe_note": "Thresholds apply to your chosen N-candle move (e.g., 3x5m or 2x15m). Recalibrate per timeframe/time-bucket.",
    "benchmark_note": "Benchmarks use NIFTY sectoral indices where available; some fine-grained sectors map to the nearest index proxy.",
    "source_note": "Heuristic defaults to bootstrap training; calibrate from your own historical data."
  },
  "sector_configs": {
    "BANK": { "benchmark_index": "NIFTY BANK", "volatility_profile": "High", "strong": 1.10, "normal": 0.40 },

    "IT": { "benchmark_index": "NIFTY IT", "volatility_profile": "Medium-High", "strong": 0.90, "normal": 0.35 },

    "FMCG": { "benchmark_index": "NIFTY FMCG", "volatility_profile": "Low", "strong": 0.55, "normal": 0.20 },

    "METAL": { "benchmark_index": "NIFTY METAL", "volatility_profile": "Very High", "strong": 1.30, "normal": 0.50 },

    "AUTO": { "benchmark_index": "NIFTY AUTO", "volatility_profile": "Medium", "strong": 0.80, "normal": 0.30 },

    "PHARMA": { "benchmark_index": "NIFTY PHARMA", "volatility_profile": "Medium-Low", "strong": 0.65, "normal": 0.25 },

    "REALTY": { "benchmark_index": "NIFTY REALTY", "volatility_profile": "High", "strong": 1.00, "normal": 0.40 },

    "CONSUMER_DURABLES": { "benchmark_index": "NIFTY CONSUMER DURABLES", "volatility_profile": "Medium", "strong": 0.75, "normal": 0.28 },

    "OIL_AND_GAS": { "benchmark_index": "NIFTY OIL AND GAS", "volatility_profile": "Medium-High", "strong": 0.95, "normal": 0.35 },

    "FINANCIAL_SERVICES": { "benchmark_index": "NIFTY FINANCIAL SERVICES", "volatility_profile": "Medium-High", "strong": 0.95, "normal": 0.35 },

    "PRIVATE_BANKS": { "benchmark_index": "NIFTY BANK", "volatility_profile": "High", "strong": 1.05, "normal": 0.38 },

    "PSU_BANKS": { "benchmark_index": "NIFTY PSU BANK", "volatility_profile": "High", "strong": 1.15, "normal": 0.45 },

    "LARGE_NBFCS": { "benchmark_index": "NIFTY FINANCIAL SERVICES", "volatility_profile": "Medium-High", "strong": 1.00, "normal": 0.36 },

    "LIFE_INSURANCE": { "benchmark_index": "NIFTY FINANCIAL SERVICES", "volatility_profile": "Medium", "strong": 0.70, "normal": 0.25 },

    "GENERAL_INSURANCE": { "benchmark_index": "NIFTY FINANCIAL SERVICES", "volatility_profile": "Medium-High", "strong": 0.80, "normal": 0.28 },

    "EXCHANGES_AND_BROKERS": { "benchmark_index": "NIFTY FINANCIAL SERVICES", "volatility_profile": "High", "strong": 1.10, "normal": 0.40 },

    "ASSET_MANAGERS_AMC": { "benchmark_index": "NIFTY FINANCIAL SERVICES", "volatility_profile": "Medium", "strong": 0.85, "normal": 0.30 },

    "CARDS_AND_PAYMENTS": { "benchmark_index": "NIFTY FINANCIAL SERVICES", "volatility_profile": "High", "strong": 1.20, "normal": 0.45 },

    "IT_SERVICES": { "benchmark_index": "NIFTY IT", "volatility_profile": "Medium-High", "strong": 0.90, "normal": 0.35 },

    "PLATFORM_AND_INTERNET": { "benchmark_index": "NIFTY 50", "volatility_profile": "High", "strong": 1.10, "normal": 0.40 },

    "AUTO_OEM_4W_2W": { "benchmark_index": "NIFTY AUTO", "volatility_profile": "Medium", "strong": 0.80, "normal": 0.30 },

    "AUTO_ANCILLARIES": { "benchmark_index": "NIFTY AUTO", "volatility_profile": "Medium-High", "strong": 0.90, "normal": 0.35 },

    "METALS_PRIVATE": { "benchmark_index": "NIFTY METAL", "volatility_profile": "Very High", "strong": 1.30, "normal": 0.50 },

    "METALS_PSU_AND_MINING": { "benchmark_index": "NIFTY METAL", "volatility_profile": "High", "strong": 1.15, "normal": 0.45 },

    "OIL_GAS_PSU": { "benchmark_index": "NIFTY OIL AND GAS", "volatility_profile": "Medium-High", "strong": 1.00, "normal": 0.38 },

    "OIL_GAS_PRIVATE_CITYGAS": { "benchmark_index": "NIFTY OIL AND GAS", "volatility_profile": "Medium", "strong": 0.90, "normal": 0.33 },

    "POWER_GENERATION_AND_UTILITIES": { "benchmark_index": "NIFTY 50", "volatility_profile": "Medium", "strong": 0.80, "normal": 0.30 },

    "FMCG_STAPLES": { "benchmark_index": "NIFTY FMCG", "volatility_profile": "Low", "strong": 0.55, "normal": 0.20 },

    "DISCRETIONARY_AND_RETAIL": { "benchmark_index": "NIFTY 50", "volatility_profile": "Medium-High", "strong": 0.90, "normal": 0.35 },

    "PHARMA_LARGE": { "benchmark_index": "NIFTY PHARMA", "volatility_profile": "Medium-Low", "strong": 0.65, "normal": 0.25 },

    "HOSPITALS_AND_DIAGNOSTICS": { "benchmark_index": "NIFTY PHARMA", "volatility_profile": "Medium", "strong": 0.80, "normal": 0.30 },

    "CAPITAL_GOODS_AND_ENGINEERING": { "benchmark_index": "NIFTY 50", "volatility_profile": "Medium-High", "strong": 0.95, "normal": 0.35 },

    "INFRA_AND_CONSTRUCTION": { "benchmark_index": "NIFTY 50", "volatility_profile": "Medium-High", "strong": 0.90, "normal": 0.35 },

    "CEMENT_AND_BUILDING_MATERIALS": { "benchmark_index": "NIFTY 50", "volatility_profile": "Medium-High", "strong": 0.90, "normal": 0.35 },

    "TELECOM": { "benchmark_index": "NIFTY 50", "volatility_profile": "Medium-High", "strong": 0.90, "normal": 0.35 },

    "TRAVEL_HOTELS_AND_LEISURE": { "benchmark_index": "NIFTY 50", "volatility_profile": "High", "strong": 1.00, "normal": 0.40 },

    "DEFAULT": { "benchmark_index": "NIFTY 50", "volatility_profile": "Medium", "strong": 0.75, "normal": 0.25 }
  }
}
