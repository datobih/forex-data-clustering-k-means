"""
Forex Data Collection Module
Handles fetching and storage of high-frequency forex data
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Dict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForexDataCollector:
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(self, pairs: List[str], interval: str = '5m',
                   start_date: str = '2019-01-01', 
                   end_date: str = '2024-12-31') -> Dict[str, pd.DataFrame]:
        
        # For 1-minute data, use period instead of date range
        if interval == '1m':
            logger.info(f'Fetching 1-minute data for {len(pairs)} pairs (last 7 days)')
        else:
            logger.info(f'Fetching data for {len(pairs)} pairs from {start_date} to {end_date}')
        
        data_dict = {}
        for pair in pairs:
            logger.info(f'Downloading {pair}...')
            try:
                ticker = yf.Ticker(pair)
                
                # Use period for 1-minute data, dates for others
                if interval == '1m':
                    df = ticker.history(interval=interval, period='7d')
                else:
                    df = ticker.history(interval=interval, start=start_date, end=end_date)
                
                if not df.empty:
                    df.columns = df.columns.str.lower()
                    data_dict[pair] = df
                    logger.info(f'{pair}: {len(df)} bars downloaded')
                    
                    # Save to disk
                    self._save_data(pair, df, interval)
                else:
                    logger.warning(f'No data returned for {pair}')
                    
            except Exception as e:
                logger.error(f'Error downloading {pair}: {str(e)}')
        
        return data_dict
    
    def _save_data(self, pair: str, df: pd.DataFrame, interval: str):
        filename = f'{pair.replace("=", "_")}_{interval}.pkl'
        filepath = self.data_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)
        
        logger.info(f'Saved {pair} data to {filepath}')
    
    def load_data(self, pair: str, interval: str = '5m') -> pd.DataFrame:
        filename = f'{pair.replace("=", "_")}_{interval}.pkl'
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f'Data file not found: {filepath}')
            return pd.DataFrame()
        
        with open(filepath, 'rb') as f:
            df = pickle.load(f)
        
        logger.info(f'Loaded {pair} data: {len(df)} bars')
        return df
    
    def load_all_pairs(self, pairs: List[str], interval: str = '5m') -> Dict[str, pd.DataFrame]:
        data_dict = {}
        for pair in pairs:
            df = self.load_data(pair, interval)
            if not df.empty:
                data_dict[pair] = df
        
        return data_dict
