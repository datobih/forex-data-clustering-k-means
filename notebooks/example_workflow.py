# Algorithmic Pattern Discovery - Example

This notebook demonstrates the pattern discovery workflow.

## 1. Setup
import pandas as pd
import numpy as np
from src.data_ingestion.forex_data import ForexDataCollector
from src.features.microstructure import MicrostructureFeatures
from src.models.unsupervised import PatternDiscovery

## 2. Collect Data
collector = ForexDataCollector()
data = collector.fetch_data(['EURUSD=X'], '5m', '2024-01-01', '2024-12-31')

## 3. Extract Features
feature_engine = MicrostructureFeatures()
features = feature_engine.build_feature_matrix(data['EURUSD=X'])
print(f"Features shape: {features.shape}")

## 4. Discover Patterns
discovery = PatternDiscovery()
patterns = discovery.discover_patterns(features)
print(f"Discovered {len(set(patterns))} unique patterns")

## 5. Analyze Results
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.scatter(range(len(patterns)), patterns, alpha=0.5)
plt.title('Pattern Clusters Over Time')
plt.xlabel('Time')
plt.ylabel('Cluster ID')
plt.show()
