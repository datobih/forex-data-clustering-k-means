# Pattern evaluation and profitability analysis
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class PatternEvaluator:
    """Evaluates pattern profitability by analyzing forward returns"""
    
    def __init__(self, atr_stop_multiplier=2.0, atr_target_multiplier=3.0):
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_target_multiplier = atr_target_multiplier
    
    def evaluate_patterns(self, data_dict: Dict[str, pd.DataFrame], 
                         pattern_labels: np.ndarray,
                         features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate profitability of each pattern cluster
        
        Args:
            data_dict: Dictionary of {pair: dataframe} with OHLCV data
            pattern_labels: Array of cluster labels for each bar
            features_df: Feature dataframe with index matching pattern_labels
            
        Returns:
            DataFrame with profitability metrics per cluster
        """
        logger.info('Evaluating pattern profitability...')
        
        # Reconstruct data with pattern labels
        all_data = []
        start_idx = 0
        
        for pair, df in data_dict.items():
            df_copy = df.copy()
            end_idx = start_idx + len(df)
            df_copy['pattern'] = pattern_labels[start_idx:end_idx]
            df_copy['pair'] = pair
            all_data.append(df_copy)
            start_idx = end_idx
        
        combined_df = pd.concat(all_data, axis=0)
        
        # Standardize column names to capitalized (yfinance uses lowercase)
        # But preserve 'pattern' and 'pair' columns
        rename_map = {col: col.capitalize() for col in combined_df.columns if col not in ['pattern', 'pair']}
        combined_df = combined_df.rename(columns=rename_map)
        
        # Calculate forward returns at multiple horizons
        logger.info('Calculating forward returns...')
        for horizon in [5, 10, 20, 30, 60]:  # 5min, 10min, 20min, 30min, 1hr ahead
            combined_df[f'forward_return_{horizon}'] = combined_df.groupby('pair')['Close'].pct_change(horizon).shift(-horizon)
        
        # Calculate ATR for risk-adjusted returns
        combined_df['high_low'] = combined_df['High'] - combined_df['Low']
        combined_df['high_close'] = abs(combined_df['High'] - combined_df['Close'].shift(1))
        combined_df['low_close'] = abs(combined_df['Low'] - combined_df['Close'].shift(1))
        combined_df['atr'] = combined_df[['high_low', 'high_close', 'low_close']].max(axis=1).rolling(14).mean()
        
        # Evaluate each cluster
        results = []
        
        for cluster in sorted(combined_df['pattern'].unique()):
            cluster_data = combined_df[combined_df['pattern'] == cluster].copy()
            
            metrics = {
                'cluster': cluster,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(combined_df) * 100
            }
            
            # Forward return analysis
            for horizon in [5, 10, 20, 30, 60]:
                returns = cluster_data[f'forward_return_{horizon}'].dropna()
                if len(returns) > 0:
                    metrics[f'mean_return_{horizon}m'] = returns.mean() * 100  # in %
                    metrics[f'win_rate_{horizon}m'] = (returns > 0).sum() / len(returns) * 100
                    metrics[f'sharpe_{horizon}m'] = returns.mean() / returns.std() if returns.std() > 0 else 0
                    
                    # Risk-reward using ATR
                    avg_atr = cluster_data['atr'].mean()
                    if avg_atr > 0:
                        metrics[f'return_per_atr_{horizon}m'] = returns.mean() / avg_atr
            
            # Trade simulation with stop loss and take profit
            long_trades = self._simulate_trades(cluster_data, direction='long')
            short_trades = self._simulate_trades(cluster_data, direction='short')
            
            metrics['long_win_rate'] = long_trades['win_rate']
            metrics['long_profit_factor'] = long_trades['profit_factor']
            metrics['long_avg_win'] = long_trades['avg_win']
            metrics['long_avg_loss'] = long_trades['avg_loss']
            
            metrics['short_win_rate'] = short_trades['win_rate']
            metrics['short_profit_factor'] = short_trades['profit_factor']
            metrics['short_avg_win'] = short_trades['avg_win']
            metrics['short_avg_loss'] = short_trades['avg_loss']
            
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Rank clusters by profitability
        results_df['long_score'] = (results_df['long_win_rate'] * results_df['long_profit_factor']).fillna(0)
        results_df['short_score'] = (results_df['short_win_rate'] * results_df['short_profit_factor']).fillna(0)
        results_df['best_direction'] = results_df.apply(
            lambda x: 'LONG' if x['long_score'] > x['short_score'] else 'SHORT', axis=1
        )
        results_df['overall_score'] = results_df[['long_score', 'short_score']].max(axis=1)
        
        results_df = results_df.sort_values('overall_score', ascending=False)
        
        return results_df
    
    def _simulate_trades(self, data: pd.DataFrame, direction: str) -> Dict:
        """Simulate trades with ATR-based stops and targets"""
        
        if len(data) < 20 or data['atr'].isna().all():
            return {'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0}
        
        wins = []
        losses = []
        
        for idx in data.index[:-60]:  # Need future data to simulate trade
            try:
                entry_price = data.loc[idx, 'Close']
                atr = data.loc[idx, 'atr']
                
                if pd.isna(atr) or atr == 0:
                    continue
                
                # Set stop loss and take profit based on ATR
                if direction == 'long':
                    stop_loss = entry_price - (atr * self.atr_stop_multiplier)
                    take_profit = entry_price + (atr * self.atr_target_multiplier)
                else:  # short
                    stop_loss = entry_price + (atr * self.atr_stop_multiplier)
                    take_profit = entry_price - (atr * self.atr_target_multiplier)
                
                # Check next 60 bars for stop/target hit
                future_idx = data.index[data.index.get_loc(idx)+1:data.index.get_loc(idx)+61]
                future_data = data.loc[future_idx]
                
                if len(future_data) == 0:
                    continue
                
                # Check if stop or target hit
                hit_target = False
                hit_stop = False
                
                if direction == 'long':
                    hit_target = (future_data['High'] >= take_profit).any()
                    hit_stop = (future_data['Low'] <= stop_loss).any()
                    
                    if hit_target and hit_stop:
                        # Check which happened first
                        target_bar = future_data[future_data['High'] >= take_profit].index[0]
                        stop_bar = future_data[future_data['Low'] <= stop_loss].index[0]
                        if target_bar < stop_bar:
                            wins.append((take_profit - entry_price) / entry_price)
                        else:
                            losses.append((entry_price - stop_loss) / entry_price)
                    elif hit_target:
                        wins.append((take_profit - entry_price) / entry_price)
                    elif hit_stop:
                        losses.append((entry_price - stop_loss) / entry_price)
                else:  # short
                    hit_target = (future_data['Low'] <= take_profit).any()
                    hit_stop = (future_data['High'] >= stop_loss).any()
                    
                    if hit_target and hit_stop:
                        target_bar = future_data[future_data['Low'] <= take_profit].index[0]
                        stop_bar = future_data[future_data['High'] >= stop_loss].index[0]
                        if target_bar < stop_bar:
                            wins.append((entry_price - take_profit) / entry_price)
                        else:
                            losses.append((stop_loss - entry_price) / entry_price)
                    elif hit_target:
                        wins.append((entry_price - take_profit) / entry_price)
                    elif hit_stop:
                        losses.append((stop_loss - entry_price) / entry_price)
            except:
                continue
        
        total_trades = len(wins) + len(losses)
        if total_trades == 0:
            return {'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0}
        
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else (float('inf') if total_wins > 0 else 0)
        
        avg_win = np.mean(wins) * 100 if wins else 0  # in %
        avg_loss = np.mean(losses) * 100 if losses else 0  # in %
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def get_tradeable_patterns(self, evaluation_df: pd.DataFrame, 
                              min_win_rate: float = 55.0,
                              min_profit_factor: float = 1.5,
                              min_trades: int = 50) -> pd.DataFrame:
        """
        Filter patterns that meet profitability criteria
        
        Args:
            evaluation_df: DataFrame from evaluate_patterns()
            min_win_rate: Minimum win rate percentage (default 55%)
            min_profit_factor: Minimum profit factor (default 1.5)
            min_trades: Minimum number of occurrences (default 50)
            
        Returns:
            Filtered DataFrame with only tradeable patterns
        """
        tradeable = evaluation_df[
            (evaluation_df['count'] >= min_trades) &
            (
                ((evaluation_df['long_win_rate'] >= min_win_rate) & 
                 (evaluation_df['long_profit_factor'] >= min_profit_factor)) |
                ((evaluation_df['short_win_rate'] >= min_win_rate) & 
                 (evaluation_df['short_profit_factor'] >= min_profit_factor))
            )
        ].copy()
        
        logger.info(f'Found {len(tradeable)} tradeable patterns out of {len(evaluation_df)} total patterns')
        
        return tradeable
