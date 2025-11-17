#!/usr/bin/env python
"""
Quickstart script for Algorithmic Pattern Discovery System
Run this to test the basic workflow
"""

import logging
from src.data_ingestion.forex_data import ForexDataCollector
from src.features.microstructure import MicrostructureFeatures
from src.models.unsupervised import PatternDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quickstart():
    """Quick demonstration of the pattern discovery workflow"""
    
    print("=" * 60)
    print("  Algorithmic Pattern Discovery System - Quickstart")
    print("=" * 60)
    
    # Collect a small sample of data
    print("\n[1/3] Collecting sample data...")
    collector = ForexDataCollector()
    
    # Use a shorter period for quick testing
    data = collector.fetch_data(
        pairs=['EURUSD=X'],
        interval='1h',
        start_date='2024-11-01',
        end_date='2024-11-15'
    )
    
    if not data:
        print("  No data collected. Check your internet connection.")
        return
    
    pair = list(data.keys())[0]
    df = data[pair]
    print(f"  Collected {len(df)} bars for {pair}")
    
    # Extract features
    print("\n[2/3] Extracting microstructure features...")
    feature_engine = MicrostructureFeatures()
    features = feature_engine.build_feature_matrix(df)
    print(f"  Extracted {features.shape[1]} features")
    
    # Discover patterns
    print("\n[3/3] Discovering patterns with unsupervised learning...")
    discovery = PatternDiscovery()
    patterns = discovery.discover_patterns(features)
    unique_patterns = len(set(patterns))
    print(f"  Discovered {unique_patterns} unique pattern clusters")
    
    print("\n" + "=" * 60)
    print("  Quickstart complete!")
    print("  Next steps:")
    print("    - Review main.py for full workflow")
    print("    - Check notebooks/ for analysis examples")
    print("    - Modify config/config.yaml for your needs")
    print("=" * 60)

if __name__ == '__main__':
    quickstart()
