# Feature engineering module
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MicrostructureFeatures:
    def __init__(self):
        pass
    
    # === PRICE FEATURES ===
    def calculate_price_velocity(self, df, periods=5):
        """Rate of price change"""
        return df['close'].diff(periods) / periods
    
    def calculate_price_acceleration(self, df, periods=5):
        """Second derivative of price"""
        velocity = self.calculate_price_velocity(df, periods)
        return velocity.diff(periods) / periods
    
    def calculate_returns(self, df, periods=[1, 5, 10, 20]):
        """Multiple timeframe returns"""
        returns = {}
        for p in periods:
            returns[f'return_{p}'] = df['close'].pct_change(p)
        return pd.DataFrame(returns, index=df.index)
    
    # === VWAP FEATURES ===
    def calculate_vwap(self, df):
        """Volume-Weighted Average Price"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    def calculate_vwap_distance(self, df):
        """Normalized distance from VWAP"""
        vwap = self.calculate_vwap(df)
        return (df['close'] - vwap) / vwap
    
    # === VOLUME FEATURES ===
    def calculate_volume_features(self, df):
        """Volume-based signals"""
        features = pd.DataFrame(index=df.index)
        
        # Volume ratios
        features['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
        features['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume delta (buy vs sell approximation)
        price_change = df['close'] - df['open']
        range_val = df['high'] - df['low'] + 1e-10
        buy_ratio = (price_change / range_val).clip(0, 1)
        features['volume_delta'] = df['volume'] * (2 * buy_ratio - 1)
        
        # Absorption (high volume, low movement)
        features['absorption'] = (df['volume'] / df['volume'].rolling(20).mean()) / (range_val / range_val.rolling(20).mean() + 1e-10)
        
        return features
    
    # === VOLATILITY FEATURES ===
    def calculate_volatility_features(self, df):
        """Volatility and range metrics"""
        features = pd.DataFrame(index=df.index)
        
        returns = df['close'].pct_change()
        features['volatility_5'] = returns.rolling(5).std()
        features['volatility_20'] = returns.rolling(20).std()
        
        # True Range
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift(1))
        lc = abs(df['low'] - df['close'].shift(1))
        features['true_range'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        features['atr_14'] = features['true_range'].rolling(14).mean()
        
        return features
    
    # === MOMENTUM FEATURES ===
    def calculate_momentum_features(self, df):
        """Momentum and strength indicators"""
        features = pd.DataFrame(index=df.index)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Rate of Change
        features['roc_10'] = df['close'].pct_change(10) * 100
        features['roc_20'] = df['close'].pct_change(20) * 100
        
        return features
    
    # === TEMPORAL FEATURES ===
    def calculate_temporal_features(self, df):
        """Time-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Cyclical encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Day of week
        features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Session indicators
        features['asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        features['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        features['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
        features['overlap_session'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
        
        return features
    
    # === PATTERN FEATURES ===
    def calculate_pattern_features(self, df):
        """Price pattern recognition"""
        features = pd.DataFrame(index=df.index)
        
        # Distance to round numbers (psychological levels)
        for level in [1.0000, 1.0500, 1.1000, 1.1500]:
            features[f'dist_to_{level}'] = abs(df['close'] - level)
        
        # Higher highs, lower lows
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Range position (where close is within bar range)
        range_val = df['high'] - df['low'] + 1e-10
        features['range_position'] = (df['close'] - df['low']) / range_val
        
        return features
    
    def build_feature_matrix(self, df):
        """Build comprehensive feature matrix"""
        logger.info('Building comprehensive feature matrix...')
        
        features = pd.DataFrame(index=df.index)
        
        # Add all feature groups
        features['velocity'] = self.calculate_price_velocity(df)
        features['acceleration'] = self.calculate_price_acceleration(df)
        features['vwap_dist'] = self.calculate_vwap_distance(df)
        
        # Returns
        returns = self.calculate_returns(df)
        features = pd.concat([features, returns], axis=1)
        
        # Volume features
        volume_feats = self.calculate_volume_features(df)
        features = pd.concat([features, volume_feats], axis=1)
        
        # Volatility features
        vol_feats = self.calculate_volatility_features(df)
        features = pd.concat([features, vol_feats], axis=1)
        
        # Momentum features
        momentum_feats = self.calculate_momentum_features(df)
        features = pd.concat([features, momentum_feats], axis=1)
        
        # Temporal features
        temporal_feats = self.calculate_temporal_features(df)
        features = pd.concat([features, temporal_feats], axis=1)
        
        # Pattern features
        pattern_feats = self.calculate_pattern_features(df)
        features = pd.concat([features, pattern_feats], axis=1)
        
        # Clean up NaN values
        features = features.ffill().fillna(0)
        
        # Replace inf values
        features = features.replace([np.inf, -np.inf], 0)
        
        logger.info(f'Feature matrix built: {features.shape[0]} rows x {features.shape[1]} features')
        return features
