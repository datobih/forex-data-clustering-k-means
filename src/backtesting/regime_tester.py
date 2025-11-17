# Regime-independent backtesting
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RegimeTester:
    def __init__(self):
        self.regimes = {}
    
    def test_across_regimes(self, patterns, data):
        logger.info('Testing patterns across regimes...')
        results = {}
        return results
