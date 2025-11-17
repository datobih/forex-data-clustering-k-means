# GETTING STARTED

## Quick Setup

### 1. Activate Virtual Environment
`powershell
.\venv\Scripts\Activate.ps1
`

If you get an execution policy error:
`powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
`

### 2. Install Dependencies
`powershell
pip install -r requirements.txt
`

This will install:
- pandas, numpy, scikit-learn
- tensorflow, torch
- yfinance (forex data)
- hmmlearn, statsmodels
- matplotlib, seaborn, plotly
- and more...

### 3. Test the System

#### Option A: Quick Demo (Recommended First)
`powershell
python quickstart.py
`

This runs a quick demo with recent data (last 2 weeks) to verify everything works.

#### Option B: Full System
`powershell
python main.py
`

This runs the complete pipeline:
- Downloads 5 years of data for multiple pairs
- Extracts hundreds of features
- Discovers patterns with unsupervised learning
- Tests patterns across market regimes

**Note**: The full system will take time to download all historical data on first run.

## What the System Does

1. **Collects forex data** from yfinance (EURUSD, GBPUSD, etc.)
2. **Extracts microstructure features** (velocity, VWAP, volume delta, etc.)
3. **Applies unsupervised learning** (K-Means, DBSCAN, PCA)
4. **Discovers patterns** automatically from the data
5. **Validates patterns** across different market regimes

## Configuration

Edit config/config.yaml to customize:
- Currency pairs to analyze
- Data frequency (1m, 5m, 15m, 1h)
- Date ranges
- Model parameters
- Pattern detection thresholds

## Project Structure

- src/data_ingestion/ - Data collection
- src/features/ - Feature engineering
- src/models/ - ML models
- src/backtesting/ - Regime testing
- src/utils/ - Helper functions
- 
otebooks/ - Analysis examples
- data/ - Cached data
- esults/ - Discovered patterns

## Next Steps After Installation

1. Run quickstart to verify setup
2. Explore 
otebooks/example_workflow.py
3. Review discovered patterns in esults/patterns/
4. Modify configuration for your needs
5. Add custom features or models
6. Build execution logic

## Troubleshooting

**yfinance issues**: If data download fails, try:
- Check internet connection
- Try different date ranges
- Use hourly data instead of minute data

**Import errors**: Make sure you're in the virtual environment:
`powershell
.\venv\Scripts\Activate.ps1
`

**Memory issues**: If processing large datasets:
- Reduce date range in config
- Use hourly instead of minute data
- Process one pair at a time

## Support

- Check README.md for full documentation
- Review PROJECT_SUMMARY.md for architecture
- See example_workflow.py for usage patterns

---

**Ready to discover algorithmic patterns!**
