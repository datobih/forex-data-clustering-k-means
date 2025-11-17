"""
Backtest Gold Cluster 7 Pattern
High-volume breakout continuation pattern
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load Gold OHLC data and pattern clusters"""
    # Load price data
    data_path = Path('data/raw/GC_F_1m.pkl')
    with open(data_path, 'rb') as f:
        price_data = pickle.load(f)
    
    # Load cluster assignments
    clusters = pd.read_csv('results/pattern_clusters.csv', index_col=0)
    clusters.index = pd.to_datetime(clusters.index)
    
    # Merge
    df = price_data.join(clusters['pattern_cluster'], how='inner')
    
    logger.info(f"Loaded {len(df)} bars of Gold data")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df

def backtest_cluster7(df, stop_loss_pct=0.001, take_profit_pct=0.002, max_hold_bars=60):
    """
    Backtest Cluster 7 pattern with percentage-based stops
    
    Parameters:
    - stop_loss_pct: 0.1% stop loss (about $2-3 on Gold at $2600)
    - take_profit_pct: 0.2% take profit (about $5-6 on Gold)
    - max_hold_bars: Maximum 60 bars (1 hour) holding period
    """
    
    trades = []
    
    # Find all Cluster 7 signals
    cluster7_signals = df[df['pattern_cluster'] == 7].index
    
    logger.info(f"Found {len(cluster7_signals)} Cluster 7 signals to backtest")
    
    for i, signal_time in enumerate(cluster7_signals):
        if i % 20 == 0:
            logger.info(f"Processing trade {i+1}/{len(cluster7_signals)}...")
        
        # Get signal bar index
        signal_idx = df.index.get_loc(signal_time)
        
        # Entry on next bar open (realistic execution)
        if signal_idx + 1 >= len(df):
            continue
            
        entry_time = df.index[signal_idx + 1]
        entry_price = df.iloc[signal_idx + 1]['open']
        
        # Calculate stops based on entry price
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
        
        # Look forward up to max_hold_bars
        exit_time = None
        exit_price = None
        exit_reason = 'timeout'
        bars_held = 0
        
        for j in range(signal_idx + 1, min(signal_idx + 1 + max_hold_bars, len(df))):
            bar = df.iloc[j]
            bars_held = j - (signal_idx + 1)
            
            # Check stop loss hit
            if bar['low'] <= stop_loss:
                exit_time = df.index[j]
                exit_price = stop_loss
                exit_reason = 'stop_loss'
                break
            
            # Check take profit hit
            if bar['high'] >= take_profit:
                exit_time = df.index[j]
                exit_price = take_profit
                exit_reason = 'take_profit'
                break
        
        # If no exit, timeout at max_hold_bars
        if exit_time is None:
            exit_idx = min(signal_idx + 1 + max_hold_bars, len(df) - 1)
            exit_time = df.index[exit_idx]
            exit_price = df.iloc[exit_idx]['close']
            exit_reason = 'timeout'
            bars_held = max_hold_bars
        
        # Calculate trade metrics
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        
        trades.append({
            'signal_time': signal_time,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'win': 1 if pnl > 0 else 0
        })
    
    trades_df = pd.DataFrame(trades)
    return trades_df

def calculate_metrics(trades_df):
    """Calculate comprehensive backtest metrics"""
    
    total_trades = len(trades_df)
    
    if total_trades == 0:
        logger.warning("No trades executed!")
        return
    
    # Win/Loss stats
    wins = trades_df[trades_df['win'] == 1]
    losses = trades_df[trades_df['win'] == 0]
    
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
    
    # PnL stats
    total_pnl = trades_df['pnl'].sum()
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    avg_win_pct = wins['pnl_pct'].mean() if len(wins) > 0 else 0
    avg_loss_pct = losses['pnl_pct'].mean() if len(losses) > 0 else 0
    
    # Profit factor
    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Exit reason breakdown
    exit_reasons = trades_df['exit_reason'].value_counts()
    
    # Holding period stats
    avg_bars_held = trades_df['bars_held'].mean()
    
    # Print results
    print("\n" + "="*70)
    print("CLUSTER 7 BACKTEST RESULTS - GOLD (XAUUSD)")
    print("="*70)
    print(f"\nTOTAL TRADES: {total_trades}")
    print(f"Date Range: {trades_df['signal_time'].min()} to {trades_df['signal_time'].max()}")
    print(f"\nWIN/LOSS STATISTICS:")
    print(f"  Wins: {num_wins} ({win_rate:.1f}%)")
    print(f"  Losses: {num_losses} ({100-win_rate:.1f}%)")
    print(f"\nPROFIT/LOSS:")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Average Win: ${avg_win:.2f} ({avg_win_pct:.2f}%)")
    print(f"  Average Loss: ${avg_loss:.2f} ({avg_loss_pct:.2f}%)")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"\nEXIT BREAKDOWN:")
    for reason, count in exit_reasons.items():
        pct = (count / total_trades) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print(f"\nAVERAGE HOLDING PERIOD: {avg_bars_held:.1f} bars")
    print("="*70)
    
    # Compare to pattern evaluation metrics
    print(f"\nCOMPARISON TO PATTERN EVALUATION:")
    print(f"  Pattern Eval Win Rate: 70.4%")
    print(f"  Backtest Win Rate: {win_rate:.1f}%")
    print(f"  Pattern Eval Profit Factor: 3.90")
    print(f"  Backtest Profit Factor: {profit_factor:.2f}")
    print("="*70 + "\n")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def main():
    """Run Cluster 7 backtest"""
    
    logger.info("Starting Cluster 7 Gold backtest...")
    
    # Load data
    df = load_data()
    
    # Run backtest with Gold-appropriate parameters
    # 0.1% stop = ~$2.60 at $2600 Gold price
    # 0.2% target = ~$5.20 at $2600 Gold price
    trades_df = backtest_cluster7(
        df,
        stop_loss_pct=0.001,      # 0.1% stop loss
        take_profit_pct=0.002,    # 0.2% take profit (2:1 reward/risk)
        max_hold_bars=60          # 60 minute timeout
    )
    
    # Save trades
    output_path = Path('results/backtest_cluster7_gold_trades.csv')
    trades_df.to_csv(output_path, index=False)
    logger.info(f"Saved trade details to {output_path}")
    
    # Calculate and display metrics
    metrics = calculate_metrics(trades_df)
    
    # Test with tighter stops
    logger.info("\n\n=== TESTING TIGHTER STOPS (0.05% / 0.15%) ===")
    trades_tight = backtest_cluster7(
        df,
        stop_loss_pct=0.0005,     # 0.05% stop (~$1.30)
        take_profit_pct=0.0015,   # 0.15% target (~$3.90)
        max_hold_bars=60
    )
    metrics_tight = calculate_metrics(trades_tight)
    
    # Test with wider stops
    logger.info("\n\n=== TESTING WIDER STOPS (0.15% / 0.30%) ===")
    trades_wide = backtest_cluster7(
        df,
        stop_loss_pct=0.0015,     # 0.15% stop (~$3.90)
        take_profit_pct=0.003,    # 0.30% target (~$7.80)
        max_hold_bars=60
    )
    metrics_wide = calculate_metrics(trades_wide)
    
    logger.info("\nBacktest complete!")

if __name__ == '__main__':
    main()
