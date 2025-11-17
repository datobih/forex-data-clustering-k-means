# Algorithmic Pattern Discovery System

An unsupervised machine learning system for discovering and backtesting trading patterns in financial markets.

## Overview

Automatically discovers hidden patterns in market microstructure data using K-Means clustering and 33 engineered features. Evaluates pattern profitability and validates through realistic backtesting.

## Quick Start

`ash
# Setup
python -m venv venv
.\\venv\\Scripts\\activate
pip install pandas numpy yfinance scikit-learn pyyaml

# Run
python main.py
python backtest_cluster7_gold.py
`

## Key Results

**Gold (XAUUSD) 1-minute:**
- 8 patterns discovered
- Cluster 7: High-volume breakout (141 occurrences)
- Evaluation: 70.4% win rate, 3.90 PF
- Backtest: 31.9% win rate, 0.88 PF

**Finding:** Significant gap between evaluation and backtest due to entry timing.

## Features

- 33 microstructure features (price, volume, volatility, momentum, temporal)
- Multi-asset support (forex, Gold, commodities)
- Multiple timeframes (1m, 5m, 15m)
- Realistic backtesting with stops/targets

## Known Limitations

1. Pattern evaluation uses forward returns, backtest uses stop/target
2. K-Means ignores temporal sequences  
3. Entry timing needs improvement
4. Not yet consistently profitable

## License

MIT License

## Disclaimer

Research/educational project. No guarantee of profitability. Not financial advice.
