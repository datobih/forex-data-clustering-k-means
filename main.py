# Main execution script
import logging
import pandas as pd
from src.data_ingestion.forex_data import ForexDataCollector
from src.features.microstructure import MicrostructureFeatures
from src.models.unsupervised import PatternDiscovery
from src.models.pattern_evaluator import PatternEvaluator
from src.backtesting.regime_tester import RegimeTester
from src.utils.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    logger.info('Starting Algorithmic Pattern Discovery System')
    
    # Load configuration
    config = load_config()
    
    # Step 1: Data Collection
    logger.info('Step 1: Collecting forex data...')
    collector = ForexDataCollector()
    pairs = config['data']['pairs']
    interval = config['data']['frequency']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    data = collector.fetch_data(pairs, interval, start_date, end_date)
    
    # Step 2: Feature Engineering
    logger.info('Step 2: Engineering features...')
    feature_engine = MicrostructureFeatures()
    
    # Process each pair
    all_features = {}
    for pair, df in data.items():
        features = feature_engine.build_feature_matrix(df)
        all_features[pair] = features
        logger.info(f'{pair}: {features.shape[1]} features extracted')
    
    # Step 3: Pattern Discovery
    logger.info('Step 3: Discovering patterns...')
    discovery = PatternDiscovery()
    
    # Combine features from all pairs
    combined_features = pd.concat(all_features.values(), axis=0)
    patterns = discovery.discover_patterns(combined_features)
    
    logger.info(f'Discovered {len(set(patterns))} unique patterns')
    
    # Save patterns with features
    logger.info('Saving pattern analysis...')
    pattern_df = combined_features.copy()
    pattern_df['pattern_cluster'] = patterns
    pattern_df.to_csv('results/pattern_clusters.csv', index=True)
    logger.info('Pattern clusters saved to results/pattern_clusters.csv')
    
    # Save cluster statistics
    cluster_stats = pd.DataFrame({
        'cluster': range(len(set(patterns))),
        'count': [sum(patterns == i) for i in range(len(set(patterns)))],
        'percentage': [sum(patterns == i) / len(patterns) * 100 for i in range(len(set(patterns)))]
    })
    cluster_stats.to_csv('results/cluster_summary.csv', index=False)
    logger.info('Cluster summary saved to results/cluster_summary.csv')
    logger.info('\nCluster Distribution:')
    for _, row in cluster_stats.iterrows():
        logger.info(f"  Cluster {int(row['cluster'])}: {int(row['count'])} bars ({row['percentage']:.2f}%)")
    
    # Step 4: Pattern Profitability Evaluation
    logger.info('\nStep 4: Evaluating pattern profitability...')
    evaluator = PatternEvaluator(
        atr_stop_multiplier=config['execution']['stop_loss_atr_multiple'],
        atr_target_multiplier=config['execution']['take_profit_atr_multiple']
    )
    
    evaluation_results = evaluator.evaluate_patterns(data, patterns, combined_features)
    evaluation_results.to_csv('results/pattern_profitability.csv', index=False)
    logger.info('Pattern profitability saved to results/pattern_profitability.csv')
    
    # Find tradeable patterns
    tradeable = evaluator.get_tradeable_patterns(
        evaluation_results,
        min_win_rate=55.0,
        min_profit_factor=1.5,
        min_trades=50
    )
    
    if len(tradeable) > 0:
        tradeable.to_csv('results/tradeable_patterns.csv', index=False)
        logger.info(f'\n✓ Found {len(tradeable)} TRADEABLE patterns!')
        logger.info('\nTop Tradeable Patterns:')
        for _, row in tradeable.head(5).iterrows():
            logger.info(f"\n  Cluster {int(row['cluster'])} ({row['best_direction']}):")
            logger.info(f"    Occurrences: {int(row['count'])} ({row['percentage']:.2f}%)")
            if row['best_direction'] == 'LONG':
                logger.info(f"    Win Rate: {row['long_win_rate']:.1f}%")
                logger.info(f"    Profit Factor: {row['long_profit_factor']:.2f}")
                logger.info(f"    Avg Win: {row['long_avg_win']:.3f}% | Avg Loss: {row['long_avg_loss']:.3f}%")
            else:
                logger.info(f"    Win Rate: {row['short_win_rate']:.1f}%")
                logger.info(f"    Profit Factor: {row['short_profit_factor']:.2f}")
                logger.info(f"    Avg Win: {row['short_avg_win']:.3f}% | Avg Loss: {row['short_avg_loss']:.3f}%")
    else:
        logger.warning('\n✗ No tradeable patterns found with current criteria')
        logger.info('All pattern evaluations saved to results/pattern_profitability.csv')
    
    # Step 5: Regime Testing
    logger.info('\nStep 5: Testing across market regimes...')
    tester = RegimeTester()
    results = tester.test_across_regimes(patterns, data)
    
    logger.info('\nPattern discovery complete!')
    return results

if __name__ == '__main__':
    main()
