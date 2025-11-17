import pandas as pd
import numpy as np
from datetime import datetime

class Cluster1Backtest:
    """Backtest Cluster 1 pattern on historical data"""
    
    def __init__(self, stop_pips=10, target_pips=20, risk_per_trade=0.01):
        self.stop_pips = stop_pips
        self.target_pips = target_pips
        self.risk_per_trade = risk_per_trade
        self.trades = []
    
    def is_cluster1_pattern(self, row):
        """
        Check if current bar is labeled as Cluster 1
        Uses the actual cluster assignment from pattern discovery
        """
        # Simply check if this bar was assigned to cluster 1
        return row.get('pattern_cluster', -1) == 1
    
    def simulate_trade(self, entry_idx, data, direction='long'):
        """Simulate a single trade with stop loss and take profit"""
        
        entry_row = data.iloc[entry_idx]
        entry_price = entry_row['Close']
        entry_time = data.index[entry_idx]
        
        # Convert pips to price movement
        # For JPY pairs, 1 pip = 0.01, for others 1 pip = 0.0001
        pair = data['pair'].iloc[0] if 'pair' in data.columns else ''
        pip_value = 0.01 if 'JPY' in pair else 0.0001
        
        # Set stop and target based on pips
        if direction == 'long':
            stop_loss = entry_price - (self.stop_pips * pip_value)
            take_profit = entry_price + (self.target_pips * pip_value)
        else:
            stop_loss = entry_price + (self.stop_pips * pip_value)
            take_profit = entry_price - (self.target_pips * pip_value)
        
        # Check future bars (up to 60 bars / 1 hour)
        max_bars = min(60, len(data) - entry_idx - 1)
        
        for i in range(1, max_bars + 1):
            bar = data.iloc[entry_idx + i]
            
            if direction == 'long':
                # Check if target hit
                if bar['High'] >= take_profit:
                    exit_time = data.index[entry_idx + i]
                    profit_pct = (take_profit - entry_price) / entry_price * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': take_profit,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'outcome': 'win',
                        'profit_pct': profit_pct,
                        'bars_held': i
                    }
                
                # Check if stop hit
                if bar['Low'] <= stop_loss:
                    exit_time = data.index[entry_idx + i]
                    profit_pct = (stop_loss - entry_price) / entry_price * 100
                    return {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': stop_loss,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'outcome': 'loss',
                        'profit_pct': profit_pct,
                        'bars_held': i
                    }
        
        # No exit within 60 bars - close at market
        exit_row = data.iloc[entry_idx + max_bars]
        exit_price = exit_row['Close']
        exit_time = data.index[entry_idx + max_bars]
        profit_pct = (exit_price - entry_price) / entry_price * 100
        
        return {
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'direction': direction,
            'outcome': 'timeout',
            'profit_pct': profit_pct,
            'bars_held': max_bars
        }
    
    def run_backtest(self, data_with_features):
        """Run backtest on data with features"""
        
        print("=" * 80)
        print("CLUSTER 1 BACKTEST")
        print("=" * 80)
        print(f"\nData Range: {data_with_features.index[0]} to {data_with_features.index[-1]}")
        print(f"Total Bars: {len(data_with_features)}")
        print(f"Stop Loss: {self.stop_pips} pips")
        print(f"Take Profit: {self.target_pips} pips")
        print("\nScanning for Cluster 1 patterns...")
        print("-" * 80)
        
        self.trades = []
        in_trade = False
        last_exit_idx = 0
        
        for idx in range(20, len(data_with_features) - 60):  # Need history and future
            
            # Skip if still in a trade
            if in_trade and idx < last_exit_idx:
                continue
            
            row = data_with_features.iloc[idx]
            
            # Check if this bar matches Cluster 1 pattern
            if self.is_cluster1_pattern(row):
                # Execute trade
                trade = self.simulate_trade(idx, data_with_features, direction='long')
                
                if trade:
                    self.trades.append(trade)
                    in_trade = True
                    
                    # Find the exit index
                    exit_time = trade['exit_time']
                    last_exit_idx = data_with_features.index.get_loc(exit_time)
                    
                    # Print trade
                    outcome_symbol = "✓" if trade['outcome'] == 'win' else "✗" if trade['outcome'] == 'loss' else "○"
                    print(f"{outcome_symbol} Trade #{len(self.trades)}: {trade['entry_time']} → {trade['exit_time']} | "
                          f"{trade['outcome'].upper():8s} | {trade['profit_pct']:+.3f}% | {trade['bars_held']} bars")
                    
                    in_trade = False
        
        # Calculate statistics
        if len(self.trades) > 0:
            self.print_statistics()
        else:
            print("\n⚠ No Cluster 1 patterns detected in this dataset")
        
        return pd.DataFrame(self.trades)
    
    def print_statistics(self):
        """Print backtest statistics"""
        
        df = pd.DataFrame(self.trades)
        
        wins = df[df['outcome'] == 'win']
        losses = df[df['outcome'] == 'loss']
        timeouts = df[df['outcome'] == 'timeout']
        
        total_trades = len(df)
        win_count = len(wins)
        loss_count = len(losses)
        timeout_count = len(timeouts)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = wins['profit_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['profit_pct'].mean() if len(losses) > 0 else 0
        
        total_profit = df['profit_pct'].sum()
        
        gross_wins = wins['profit_pct'].sum() if len(wins) > 0 else 0
        gross_losses = abs(losses['profit_pct'].sum()) if len(losses) > 0 else 0
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf')
        
        avg_bars_held = df['bars_held'].mean()
        
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        
        print(f"\nTotal Trades: {total_trades}")
        print(f"  Wins:     {win_count} ({win_count/total_trades*100:.1f}%)")
        print(f"  Losses:   {loss_count} ({loss_count/total_trades*100:.1f}%)")
        print(f"  Timeouts: {timeout_count} ({timeout_count/total_trades*100:.1f}%)")
        
        print(f"\nWin Rate: {win_rate:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        print(f"\nAverage Win: {avg_win:.3f}%")
        print(f"Average Loss: {avg_loss:.3f}%")
        print(f"Average Profit per Trade: {total_profit/total_trades:.3f}%")
        
        print(f"\nTotal Cumulative Return: {total_profit:.2f}%")
        print(f"Average Bars Held: {avg_bars_held:.1f} bars ({avg_bars_held:.0f} minutes)")
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        print(f"\nExpectancy: {expectancy:.3f}% per trade")
        
        print("\n" + "=" * 80)


def main():
    print("Loading data...")
    
    # Load the pattern clusters data (has features + cluster labels)
    df = pd.read_csv('results/pattern_clusters.csv', index_col=0, parse_dates=True)
    
    # We need to add OHLC data back for the backtest
    # Load raw data
    import pickle
    pairs = ['EURUSD_X', 'GBPUSD_X', 'USDJPY_X', 'AUDUSD_X']
    
    all_data = []
    
    for pair in pairs:
        try:
            with open(f'data/raw/{pair}_1m.pkl', 'rb') as f:
                price_data = pickle.load(f)
            
            # Capitalize columns
            price_data.columns = price_data.columns.str.capitalize()
            
            # Get features for this pair's timeframe
            pair_features = df.loc[price_data.index].copy()
            
            # Combine OHLC with features
            combined = pd.concat([price_data[['Open', 'High', 'Low', 'Close', 'Volume']], pair_features], axis=1)
            combined['pair'] = pair
            
            all_data.append(combined)
            
            print(f"✓ Loaded {pair}: {len(combined)} bars")
        except Exception as e:
            print(f"✗ Error loading {pair}: {e}")
    
    if len(all_data) == 0:
        print("Error: No data loaded!")
        return
    
    # Run backtest on each pair separately
    for data in all_data:
        pair = data['pair'].iloc[0]
        print(f"\n{'='*80}")
        print(f"BACKTESTING {pair}")
        print(f"{'='*80}")
        
        backtest = Cluster1Backtest(stop_pips=10, target_pips=20)
        trades_df = backtest.run_backtest(data)
        
        if len(trades_df) > 0:
            # Save trades
            trades_df.to_csv(f'results/backtest_{pair}_trades.csv', index=False)
            print(f"\n✓ Trades saved to results/backtest_{pair}_trades.csv")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
