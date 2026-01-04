"""
Synthetic Ship Engine Data Generator

This module generates synthetic ship engine data that matches the statistical
properties observed in the original Ahlgren & Thern (2018) research.

Based on statistical analysis from the ML-dyn repository notebooks:
- Data period: ~10 months (Feb 2014 - Dec 2014)
- Sample interval: 15 minutes (900 seconds)
- ~30,000 samples per feature

Feature Statistics (from original data):
- Engine RPM: 0-760 RPM range, bimodal distribution (0 or ~750)
- Fuel Rack Position: 0-90% range
- Exhaust Temperature: 70-480°C range
- Turbocharger RPM: 0-24,000 RPM range
- Fuel Oil Flow: 0-2.7 m³/h range (target variable)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from datetime import datetime, timedelta


class ShipEngineDataGenerator:
    """Generate synthetic ship engine sensor data."""

    # Statistical parameters derived from original research
    ENGINE_PARAMS = {
        # Auxiliary Engines (AE1-4)
        'ae_rpm': {'min': 0, 'max': 760, 'mean': 350, 'std': 370},
        'ae_frp': {'min': 0, 'max': 50, 'mean': 12, 'std': 11},
        'ae_exh_t': {'min': 70, 'max': 460, 'mean': 200, 'std': 145},
        'ae_tc_rpm': {'min': 0, 'max': 25000, 'mean': 7000, 'std': 8000},

        # Main Engines (ME1-4)
        'me_rpm': {'min': 0, 'max': 500, 'mean': 130, 'std': 180},
        'me_frp': {'min': 0, 'max': 90, 'mean': 12, 'std': 19},
        'me_exh_t': {'min': 70, 'max': 485, 'mean': 185, 'std': 155},
        'me_tc_rpm': {'min': 0, 'max': 22000, 'mean': 4500, 'std': 6800},
    }

    # Fuel consumption model coefficients (calibrated to match original ~0-0.75 m³/h range)
    FUEL_COEFFICIENTS = {
        'ae_rpm': 0.0003,
        'ae_frp': 0.004,
        'ae_exh_t': 0.0002,
        'ae_tc_rpm': 0.000005,
        'me_rpm': 0.0006,
        'me_frp': 0.003,
        'me_exh_t': 0.00015,
        'me_tc_rpm': 0.000004,
        'intercept': 0.02,
        'noise_std': 0.02
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def _generate_bimodal_engine_signal(
        self,
        n_samples: int,
        params: Dict,
        operating_fraction: float = 0.6
    ) -> np.ndarray:
        """
        Generate bimodal engine signal (engine on/off behavior).

        Ship engines often operate in bimodal states - either running
        at operational speed or turned off (especially auxiliary engines).
        """
        # Determine which samples are "engine on"
        is_operating = self.rng.random(n_samples) < operating_fraction

        signal = np.zeros(n_samples)
        n_operating = is_operating.sum()

        # For operating samples, generate from truncated normal
        if n_operating > 0:
            operating_values = self.rng.normal(
                params['max'] * 0.9,
                params['std'] * 0.3,
                n_operating
            )
            operating_values = np.clip(operating_values, params['min'], params['max'])
            signal[is_operating] = operating_values

        # Add temporal autocorrelation (engines don't switch instantly)
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        signal = np.convolve(signal, kernel, mode='same')

        return np.clip(signal, params['min'], params['max'])

    def _generate_engine_features(
        self,
        n_samples: int,
        engine_prefix: str,
        engine_type: str
    ) -> Dict[str, np.ndarray]:
        """Generate all features for a single engine."""

        type_prefix = 'ae' if engine_type == 'auxiliary' else 'me'

        # Operating fraction varies by engine type
        op_frac = 0.55 if engine_type == 'auxiliary' else 0.4

        # Generate RPM (bimodal - on/off behavior)
        rpm = self._generate_bimodal_engine_signal(
            n_samples,
            self.ENGINE_PARAMS[f'{type_prefix}_rpm'],
            operating_fraction=op_frac
        )

        # FRP correlates with RPM (more fuel when engine running)
        frp_base = rpm / self.ENGINE_PARAMS[f'{type_prefix}_rpm']['max']
        frp = frp_base * self.ENGINE_PARAMS[f'{type_prefix}_frp']['max']
        frp += self.rng.normal(0, 2, n_samples)
        frp = np.clip(frp, 0, self.ENGINE_PARAMS[f'{type_prefix}_frp']['max'])

        # Exhaust temperature correlates with RPM (hot when running)
        exh_base = rpm / self.ENGINE_PARAMS[f'{type_prefix}_rpm']['max']
        exh_t = 75 + exh_base * 350  # 75°C idle, up to 425°C running
        exh_t += self.rng.normal(0, 30, n_samples)
        exh_t = np.clip(exh_t, 70, 485)

        # Turbocharger RPM correlates with engine RPM
        tc_rpm = rpm * 30 + self.rng.normal(0, 1000, n_samples)
        tc_rpm = np.clip(tc_rpm, 0, 25000)

        return {
            f'{engine_prefix}_rpm': rpm,
            f'{engine_prefix}_frp': frp,
            f'{engine_prefix}_exh_t': exh_t,
            f'{engine_prefix}_tc_rpm': tc_rpm
        }

    def _compute_fuel_consumption(
        self,
        features: Dict[str, np.ndarray],
        booster_group: str
    ) -> np.ndarray:
        """
        Compute fuel oil consumption based on engine features.

        Uses a nonlinear model with interactions to approximate
        real engine fuel consumption behavior.
        """
        c = self.FUEL_COEFFICIENTS
        n_samples = len(next(iter(features.values())))

        # Select engines for this booster group (1&3 or 2&4)
        if booster_group == '13':
            engines = ['ae1', 'ae3', 'me1', 'me3']
        else:
            engines = ['ae2', 'ae4', 'me2', 'me4']

        fuel = np.ones(n_samples) * c['intercept']

        for engine in engines:
            engine_type = 'ae' if engine.startswith('ae') else 'me'

            rpm = features.get(f'{engine}_rpm', np.zeros(n_samples))
            frp = features.get(f'{engine}_frp', np.zeros(n_samples))
            exh_t = features.get(f'{engine}_exh_t', np.zeros(n_samples))
            tc_rpm = features.get(f'{engine}_tc_rpm', np.zeros(n_samples))

            # Linear contributions
            fuel += c[f'{engine_type}_rpm'] * rpm
            fuel += c[f'{engine_type}_frp'] * frp
            fuel += c[f'{engine_type}_exh_t'] * exh_t * 0.001  # scaled
            fuel += c[f'{engine_type}_tc_rpm'] * tc_rpm

            # Nonlinear interaction: FRP * RPM (realistic fuel behavior)
            fuel += 0.00001 * frp * rpm

        # Add realistic noise
        noise = self.rng.normal(0, c['noise_std'], n_samples)
        fuel += noise

        return np.clip(fuel, 0, 3.0)  # Max ~3 m³/h based on original data

    def generate_dataset(
        self,
        n_samples: int = 30000,
        start_date: str = '2014-02-01',
        sample_interval_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Generate complete synthetic ship engine dataset.

        Args:
            n_samples: Number of time samples to generate
            start_date: Start date for time index
            sample_interval_minutes: Minutes between samples

        Returns:
            DataFrame with all engine features and fuel consumption targets
        """
        print(f"Generating synthetic ship engine data ({n_samples} samples)...")

        # Generate time index
        start = pd.Timestamp(start_date)
        time_index = pd.date_range(
            start=start,
            periods=n_samples,
            freq=f'{sample_interval_minutes}min'
        )

        all_features = {}

        # Generate features for all 8 engines
        engine_configs = [
            ('ae1', 'auxiliary'), ('ae2', 'auxiliary'),
            ('ae3', 'auxiliary'), ('ae4', 'auxiliary'),
            ('me1', 'main'), ('me2', 'main'),
            ('me3', 'main'), ('me4', 'main'),
        ]

        for engine_prefix, engine_type in engine_configs:
            engine_features = self._generate_engine_features(
                n_samples, engine_prefix, engine_type
            )
            all_features.update(engine_features)

        # Compute fuel consumption for both booster pump groups
        all_features['fo_booster_13'] = self._compute_fuel_consumption(
            all_features, '13'
        )
        all_features['fo_booster_24'] = self._compute_fuel_consumption(
            all_features, '24'
        )

        # Create DataFrame with time index
        df = pd.DataFrame(all_features, index=time_index)
        df.index.name = 'timestamp'

        print(f"Generated dataset shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def create_feature_combinations(self) -> List[Tuple[List[str], str]]:
        """
        Create the 15 feature combinations tested in the original paper.

        Original paper tested all combinations of:
        - rpm: Engine speed
        - frp: Fuel rack position
        - exh_t: Exhaust temperature
        - tc_rpm: Turbocharger RPM

        Returns:
            List of (feature_list, description) tuples
        """
        import itertools

        feature_types = ['rpm', 'frp', 'exh_t', 'tc_rpm']
        combinations = []

        for r in range(1, len(feature_types) + 1):
            for combo in itertools.combinations(feature_types, r):
                feature_list = list(combo)
                description = '+'.join(feature_list)
                combinations.append((feature_list, description))

        return combinations

    def prepare_experiment_data(
        self,
        df: pd.DataFrame,
        feature_types: List[str],
        engine_group: str = '13'  # '13' or '24'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X and y arrays for a specific feature combination.

        Args:
            df: Full dataset DataFrame
            feature_types: List of feature types to include (e.g., ['rpm', 'frp'])
            engine_group: Which engine group to use ('13' or '24')

        Returns:
            X: Feature matrix
            y: Target vector (fuel consumption)
        """
        # Select engines for this group
        if engine_group == '13':
            engines = ['ae1', 'ae3', 'me1', 'me3']
            target = 'fo_booster_13'
        else:
            engines = ['ae2', 'ae4', 'me2', 'me4']
            target = 'fo_booster_24'

        # Build feature column list
        feature_cols = []
        for engine in engines:
            for feat_type in feature_types:
                col_name = f'{engine}_{feat_type}'
                if col_name in df.columns:
                    feature_cols.append(col_name)

        X = df[feature_cols].values
        y = df[target].values

        return X, y, feature_cols


if __name__ == '__main__':
    # Test data generation
    generator = ShipEngineDataGenerator(seed=42)
    df = generator.generate_dataset(n_samples=1000)

    print("\nDataset Statistics:")
    print(df.describe())

    print("\nFeature Combinations:")
    combos = generator.create_feature_combinations()
    for i, (features, desc) in enumerate(combos):
        print(f"  {i+1}. {desc}")
