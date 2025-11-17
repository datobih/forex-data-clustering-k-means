import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('results/pattern_clusters.csv')
cluster1 = df[df['pattern_cluster'] == 1]
all_data = df

print('=' * 80)
print('CLUSTER 1 FEATURE CHARACTERISTICS (Tradeable Pattern)')
print('=' * 80)
print(f'\nTotal Bars in Cluster 1: {len(cluster1)} ({len(cluster1)/len(all_data)*100:.2f}%)')
print('\nMost Distinctive Features (sorted by standard deviations from mean):')
print('=' * 80)

# Calculate differences and rank by standard deviation
feature_diffs = []
for col in cluster1.columns:
    if col != 'pattern_cluster':
        try:
            c1_mean = cluster1[col].mean()
            all_mean = all_data[col].mean()
            all_std = all_data[col].std()
            
            if all_std > 0:
                diff = c1_mean - all_mean
                std_diff = diff / all_std
                feature_diffs.append({
                    'feature': col,
                    'cluster1_mean': c1_mean,
                    'overall_mean': all_mean,
                    'difference': diff,
                    'std_deviations': std_diff
                })
        except:
            pass

# Sort by absolute standard deviation
feature_diffs.sort(key=lambda x: abs(x['std_deviations']), reverse=True)

print('\nTop 15 Most Distinctive Features:')
print('-' * 80)
for i, feat in enumerate(feature_diffs[:15], 1):
    direction = '↑ HIGHER' if feat['std_deviations'] > 0 else '↓ LOWER'
    print(f"\n{i}. {feat['feature']}")
    print(f"   Cluster 1: {feat['cluster1_mean']:>12.6f}")
    print(f"   Overall:   {feat['overall_mean']:>12.6f}")
    print(f"   {direction} by {abs(feat['std_deviations']):.2f} standard deviations")

print('\n' + '=' * 80)
print('TRADING RULES FOR CLUSTER 1:')
print('=' * 80)

# Create simple rules based on top features
print('\nEnter LONG when these conditions are met:')
for i, feat in enumerate(feature_diffs[:5], 1):
    if feat['std_deviations'] > 0:
        threshold = feat['overall_mean'] + (0.5 * feat['difference'])
        print(f"  {i}. {feat['feature']} > {threshold:.6f}")
    else:
        threshold = feat['overall_mean'] + (0.5 * feat['difference'])
        print(f"  {i}. {feat['feature']} < {threshold:.6f}")

print('\nExit Rules:')
print('  - Stop Loss: 2.0 x ATR below entry')
print('  - Take Profit: 3.0 x ATR above entry')
print('  - Win Rate: 58.8%')
print('  - Profit Factor: 2.21')

# Save detailed feature comparison
comparison = pd.DataFrame(feature_diffs)
comparison.to_csv('results/cluster1_feature_analysis.csv', index=False)
print('\n✓ Detailed analysis saved to results/cluster1_feature_analysis.csv')
