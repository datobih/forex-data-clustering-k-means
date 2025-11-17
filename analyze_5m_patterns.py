import pandas as pd

df = pd.read_csv('results/pattern_profitability.csv')

print('5-MINUTE GOLD - ALL PATTERNS:\n')
print('='*70)

for _, row in df.iterrows():
    best_dir = row['best_direction']
    
    if best_dir == 'LONG':
        wr = row['long_win_rate']
        pf = row['long_profit_factor']
    else:
        wr = row['short_win_rate']
        pf = row['short_profit_factor']
    
    cluster = int(row['cluster'])
    count = int(row['count'])
    pct = row['percentage']
    
    status = '✓ PASS' if wr >= 55 and pf >= 1.5 and count >= 50 else '✗ FAIL'
    
    print(f"Cluster {cluster}: {count} bars ({pct:.1f}%) - {best_dir} {status}")
    print(f"  Win Rate: {wr:.1f}% | Profit Factor: {pf:.2f}")
    print()

print('='*70)
print('\nFILTER THRESHOLDS:')
print('Min Win Rate: 55% | Min PF: 1.5 | Min Occurrences: 50')
print('\nRESULT: No patterns meet all criteria')
